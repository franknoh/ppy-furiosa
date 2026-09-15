"""Lower semantic operations to the supported physical BF16 broadcast pipeline."""

from __future__ import annotations

from dataclasses import replace

from ppy_compiler.backend import BackendValidationError
from ppy_compiler.ir import Builder, IRFunction, IRModule, Operation, Pass, PassContext

from .compatibility import create_operation, ir_attributes
from .mapping import Mapping, Symbol
from .physical import broadcast_pipeline
from .semantic import SemanticTensor
from .tensor import Element, Memory, Tensor


class LowerBroadcast(Pass):
    name = "furiosa-lower-broadcast"

    def run(self, module: IRModule, ctx: PassContext) -> bool:
        functions = [
            function
            for function in module.functions.values()
            if any(op.name.startswith("furiosa.") for op in function.operations())
        ]
        if not functions:
            return False
        if len(functions) != 1 or len(module.functions) != 1:
            raise BackendValidationError(
                "furiosa", "source broadcast supports one device function per module"
            )
        self._lower_function(module, functions[0])
        module.require("rngd", 1)
        module.dialects.pop("furiosa", None)
        return True

    @staticmethod
    def _lower_function(module: IRModule, function: IRFunction) -> None:
        if len(function.body.blocks) != 1 or function.results or len(function.params) != 2:
            raise BackendValidationError(
                "furiosa",
                "broadcast requires a single-block void function with two tensor parameters",
                location=function.location,
            )
        block = function.body.blocks[0]
        if tuple(op.name for op in block.operations) != (
            "furiosa.broadcast",
            "furiosa.store",
            "core.ret",
        ):
            raise BackendValidationError(
                "furiosa",
                "unsupported source flow: expected broadcast followed by store",
                location=function.location,
            )
        broadcast, store, _ = block.operations
        if broadcast.operands != [block.arguments[0]] or store.operands != [
            block.arguments[1],
            broadcast.result,
        ]:
            raise BackendValidationError(
                "furiosa",
                "broadcast must read the input and store its result to the output",
                location=function.location,
            )
        source = SemanticTensor.from_ir(block.arguments[0].type)
        target = SemanticTensor.from_ir(block.arguments[1].type)
        target.check_store(source.broadcast(ir_attributes(broadcast).get("copies")))
        hidden, copies = Symbol("H"), Symbol("Copies")
        vector, replicated = Mapping((hidden,)), Mapping((copies, hidden))
        physical = (
            Tensor(Element.BF16, Memory.HBM, vector, vector),
            Tensor(Element.BF16, Memory.HBM, replicated, replicated, mutable=True),
        )
        function.params = tuple(
            (name, tensor.to_ir())
            for (name, _), tensor in zip(function.params, physical, strict=True)
        )
        for argument, tensor in zip(block.arguments, physical, strict=True):
            argument.type = tensor.to_ir()
        ir_attributes(function)["rngd.device"] = True
        ir_attributes(module)["rngd.axes"] = {"H": source.shape[0], "Copies": 256}
        _lower_broadcast(broadcast, physical[0])
        builder = Builder(location=store.location).before(store)
        create_operation(builder, "rngd.to_hbm", [store.operands[1], store.operands[0]])
        store.erase()


def _lower_broadcast(op: Operation, source: Tensor) -> None:
    builder = Builder(location=op.location).before(op)
    local = replace(source, memory=Memory.DM)
    result = replace(
        local, logical=Mapping((Symbol("Copies"), Symbol("H"))), slices=Mapping((Symbol("Copies"),))
    )
    loaded = create_operation(builder, "rngd.to_dm", op.operands, [local.to_ir()]).result
    pipeline = broadcast_pipeline(builder, loaded, local, result)
    op.result.replace_all_uses_with(pipeline.result)
    op.erase()
