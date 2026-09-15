#![feature(register_tool)]
#![register_tool(furiosa_opt)]

use furiosa_opt_std::prelude::*;

axes![Copies = 256, H = 3840];

#[device(chip = 1)]
pub fn kernel_broadcast(
    ctx: &mut Context,
    x: &HbmTensor<bf16, m![1], m![H]>,
    out: &mut HbmTensor<bf16, m![1], m![Copies, H]>,
) {
    let v0: DmTensor<bf16, m![1], m![1 # 2], m![1 # 256], m![H]> = x.to_dm(&mut ctx.tdma);
    let v1: DmTensor<bf16, m![1], m![1 # 2], m![Copies], m![H]> = ctx.main
        .begin(v0.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Copies], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    v1.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}
