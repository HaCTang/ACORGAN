import organ
from organ import ORGAN

model = ORGAN('cond_NAPro', 'mol_metrics', params={
    'CLASS_NUM': 2,
    'PRETRAIN_DIS_EPOCHS': 1,
    'PRETRAIN_GEN_EPOCHS': 250,
    'MAX_LENGTH': 100,
    'LAMBDA': 0.5,
    'EPOCH_SAVES':4
})

model.load_training_set('./data/train_NAPro.csv')
model.set_training_program(['diversity'], [2])
model.load_metrics()

model.load_prev_pretraining(ckpt='ckpt/cond_test_pretrain_ckpt')
model.organ_train(ckpt_dir='ckpt')
# model.load_prev_training(ckpt='./checkpoints/cond_test/cond_test_9.ckpt')
model.conditional_train(ckpt_dir='ckpt', gen_steps=30)

# model.load_prev_training(ckpt='ckpt/cond_test_0.ckpt')

# then generate samples
model.output_samples(10000, label_input=True, target_class=0)
model.output_samples(10000, label_input=True, target_class=1)
