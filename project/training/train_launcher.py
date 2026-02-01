from train_student import StudentTrainer
import torch
import os

dataset_paths = ['datasets/dataset_a/', 'datasets/dataset_b/', 'datasets/dataset_ab/']
datasets = {'train': 'train.csv', 'valid': 'valid.csv', 'test': 'test.csv'}
student_output_paths = ['student_a/', 'student_b/', 'student_ab/']
model_names = ['google/t5-efficient-mini', 'google/t5-efficient-small', 'google/t5-efficient-base', 'google/mt5-small', 'google/mt5-base']

for student_model_name in model_names:
    for i in range(len(dataset_paths)):
        train_dataset = dataset_paths[i] + datasets['train']
        valid_dataset = dataset_paths[i] + datasets['valid']

        train_output_path = 'models/' + student_model_name + '/' + student_output_paths[i]

        trainer = StudentTrainer(train_dataset, valid_dataset, batch_size=16, train_output_path=train_output_path, train_epochs=30, learning_rate=2e-5, student_model_name=student_model_name, dtype=torch.float16)
        trainer.train_student()

        test_dataset_a = dataset_paths[0] + datasets['test']
        test_dataset_b = dataset_paths[1] + datasets['test']

        model_path = os.path.abspath(train_output_path + 'best-model/')
        outupt_path_a = train_output_path + 'prediction_a.csv'
        outupt_path_b = train_output_path + 'prediction_b.csv'
        trainer.run_inference(test_dataset_a, model_path, outupt_path_a)
        trainer.run_inference(test_dataset_b, model_path, outupt_path_b)