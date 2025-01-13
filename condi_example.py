import organ
from organ import ORGAN

organ_params = {'CLASS_NUM': 2,
    'PRETRAIN_GEN_EPOCHS': 250, 'PRETRAIN_DIS_EPOCHS': 10, 'MAX_LENGTH': 80, 'LAMBDA_C': 0.2, "DIS_EPOCHS": 2, 'SAMPLE_NUM': 6400, 'WGAN':True}

# hyper-optimized parameters
params = {"GEN_HIDDEN_DIM": 64, "DIS_L2REG": 0.2, "DIS_EMB_DIM": 32, "DIS_FILTER_SIZES": [
    1, 2, 3, 4, 5, 7, 10, 15, 20], "DIS_NUM_FILTERS": [50, 50, 50, 50, 50, 50, 50, 75, 100], "DIS_DROPOUT": 0.75, "EPOCH_SAVES": 5}

organ_params.update(params)

model = ORGAN('try_classifier', 'mol_metrics', params=organ_params)

model.load_training_set('./data/train_NAPro.csv')
# model.set_training_program(['diversity'], [2])
# model.load_metrics()

# model.load_prev_pretraining(ckpt='ckpt/try_classifier_15_ckpt')
# model.organ_train(ckpt_dir='ckpt')
model.load_prev_training(ckpt='./ckpt/try_classifier_15.ckpt')
model.conditional_train(ckpt_dir='ckpt', gen_steps=30)

# model.load_prev_training(ckpt='ckpt/cond_NAPro_8.ckpt')

# then generate samples
# model.output_samples(10000, label_input=True, target_class=0)
# model.output_samples(10000, label_input=True, target_class=1)
