import organ
from organ import ORGAN

model = ORGAN('cond_NAPro', 'mol_metrics', params={
    'CLASS_NUM': 2,
    'PRETRAIN_DIS_EPOCHS': 1,
    'PRETRAIN_GEN_EPOCHS': 200,
    'MAX_LENGTH': 100,
    'LAMBDA': 0.5,
    'EPOCH_SAVES':5,
    "DIS_L2REG": 0.2, 
    "DIS_EMB_DIM": 32, 
    "DIS_FILTER_SIZES": [1, 2, 3, 4, 5, 8, 10, 15], 
    "DIS_NUM_FILTERS": [50, 50, 50, 50, 50, 50, 50, 75], 
    "DIS_DROPOUT": 0.75,
    "DIS_EPOCHS": 2
})

model.load_training_set('./data/train_NAPro.csv')
model.set_training_program(['diversity'], [2])
model.load_metrics()

model.load_prev_pretraining(ckpt='ckpt/cond_NAPro_pretrain_ckpt')
# model.organ_train(ckpt_dir='ckpt')
# model.load_prev_training(ckpt='./checkpoints/cond_NAPro/cond_NAPro_1.ckpt')
model.conditional_train(ckpt_dir='ckpt', gen_steps=30)

# model.load_prev_training(ckpt='ckpt/cond_NAPro_8.ckpt')

# then generate samples
model.output_samples(10000, label_input=True, target_class=0)
model.output_samples(10000, label_input=True, target_class=1)
