#[cutile::module]
pub mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    pub fn gelu_mul<const BLOCK: i32>(
        g: &Tensor<bf16, {[-1]}>, u: &Tensor<bf16, {[-1]}>,
        out: &mut Tensor<bf16, {[BLOCK]}>,
    ) {
        let block = get_tile_block_id().0;
        let x: Tile<f32, {[BLOCK]}> = convert_tile(g.partition(shape![BLOCK]).load([block]));
        let y: Tile<f32, {[BLOCK]}> = convert_tile(u.partition(shape![BLOCK]).load([block]));
        let inner = (x + x*x*x*constant(0.044715f32, shape![BLOCK]))*constant(0.7978845608028654f32, shape![BLOCK]);
        let a = (x*constant(0.5f32, shape![BLOCK]))*(tanh(inner)+constant(1.0f32, shape![BLOCK]));
        let rounded: Tile<bf16, {[BLOCK]}> = convert_tile(a);
        let rounded: Tile<f32, {[BLOCK]}> = convert_tile(rounded);
        let result: Tile<bf16, {[BLOCK]}> = convert_tile(rounded*y);
        out.store(result);
    }
    #[cutile::entry()]
    pub fn change<const BLOCK: i32>(g: &mut Tensor<bf16, {[BLOCK]}>) {
        let x: Tile<f32, {[BLOCK]}> = convert_tile(g.load());
        let changed: Tile<bf16, {[BLOCK]}> = convert_tile(x + constant(0.5f32, shape![BLOCK]));
        g.store(changed);
    }

}
