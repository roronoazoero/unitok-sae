from sae.unitok_loader import build_unitok
model = build_unitok('/scratch/work/vuh12/UniTok/unitok_tokenizer.pth', 'cpu')
print(model.encoder)  # Shows encoder structure
print(model.encoder.blocks[15])  # Check block 15 exists