from train_student import StudentTrainer
import os

dataset_paths = ['datasets/dataset_a/', 'datasets/dataset_b/', 'datasets/dataset_ab/']
datasets = {'train': 'train.csv', 'valid': 'valid.csv', 'test': 'test.csv'}
student_output_paths = ['models/student_a/', 'models/student_b/', 'models/student_ab/']

for i in range(len(dataset_paths)):
    train_dataset = dataset_paths[i] + datasets['train']
    valid_dataset = dataset_paths[i] + datasets['valid']

    trainer = StudentTrainer(train_dataset, valid_dataset, student_output_paths[i], train_epochs=2, learning_rate=2e-5, test_train=True)
    trainer.train_student()

    test_dataset_a = dataset_paths[0] + datasets['test']
    test_dataset_b = dataset_paths[1] + datasets['test']

    model_path = os.path.abspath(student_output_paths[i] + 'best-model/')
    outupt_path_a = student_output_paths[i] + 'prediction_a.csv'
    outupt_path_b = student_output_paths[i] + 'prediction_b.csv'
    trainer.run_inference(test_dataset_a, model_path, outupt_path_a)
    trainer.run_inference(test_dataset_b, model_path, outupt_path_b)