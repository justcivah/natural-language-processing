import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import sacrebleu
import numpy as np
import os
import csv

class StudentTrainer:
    def __init__(
            self,
            train_dataset_path,
            valid_dataset_path,
            train_output_path,
            student_model_name='google/t5-efficient-mini',
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            dtype=torch.float32,
            batch_size=32,
            train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            test_train=False,
        ):

        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path
        self.train_output_path = train_output_path
        self.student_model_name = student_model_name
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_train = test_train

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.student_model_name)
        self.model.to(self.device)


    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        chrf_score = sacrebleu.corpus_chrf(preds, [labels], word_order=2).score
        bleu_score = sacrebleu.corpus_bleu(preds, [labels]).score

        return {'chrfpp': chrf_score, 'bleu': bleu_score}
    

    def preprocess_function(self, examples):
            inputs = [ex for ex in examples['english']]
            targets = [ex for ex in examples['italian']]
            
            # tokenize inputs and targets
            model_inputs = self.tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
            labels = self.tokenizer(text_target=targets, max_length=160, truncation=True, padding='max_length')

            model_inputs['labels'] = labels['input_ids']
            return model_inputs
    

    def run_inference(
        self,
        test_dataset_path,
        model_path,
        output_path,
        batch_size=32,
        max_length=192,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        print('Computing predictions on test dataset ' + test_dataset_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        model.eval()

        test_dataset = load_dataset('csv', data_files=test_dataset_path)['train']

        if self.test_train:
            test_dataset = test_dataset.select(range(25))

        predictions = []
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i + batch_size]
            inputs = batch['english']

            encodings = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**encodings, max_length=max_length)

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for pred in predictions:
                writer.writerow([pred])
        
        print('Saved predictions in ' + output_path)


    def save_training_losses(self, trainer, training_losses_output_path):
        merged = {}

        for log in trainer.state.log_history:
            epoch = log.get("epoch")
            if epoch is None:
                continue

            if epoch not in merged:
                merged[epoch] = {"epoch": epoch}

            merged[epoch].update(log)

        with open(training_losses_output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "step", "loss", "eval_loss", "eval_chrfpp", "eval_bleu"])
            writer.writeheader()

            for epoch in sorted(merged):
                writer.writerow({
                    "epoch": epoch,
                    "step": merged[epoch].get("step"),
                    "loss": merged[epoch].get("loss"),
                    "eval_loss": merged[epoch].get("eval_loss"),
                    "eval_chrfpp": merged[epoch].get("eval_chrfpp"),
                    "eval_bleu": merged[epoch].get("eval_bleu"),
                })


    def train_student(self):
        # model and tokenizer
        print(f'Model on device: {self.model.device}')
        print(f'Model training using dtype: {self.model.dtype}')


        # dataset
        print('Loading datasets...')

        train_dataset = load_dataset('csv', data_files=self.train_dataset_path)['train']
        valid_dataset = load_dataset('csv', data_files=self.valid_dataset_path)['train']

        if self.test_train:
            train_dataset = train_dataset.select(range(100))
            valid_dataset = valid_dataset.select(range(25))

        print(f'Number of samples in training: {len(train_dataset)}')
        print(f'Number of samples in validation: {len(valid_dataset)}')

        sample_train = train_dataset[0]
        sample_valid = valid_dataset[0]

        print(f'Sample training input: {sample_train['english']}')
        print(f'Sample training output: {sample_train['italian']}')
        print(f'Sample validation input: {sample_valid['english']}')
        print(f'Sample validation output: {sample_valid['italian']}')


        # preprocessing
        print('Tokenizing datasets...')

        # tokenize the training and validation datasets
        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_valid = valid_dataset.map(self.preprocess_function, batched=True)


        # training config
        print('Setting trainer configuration...')

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.train_output_path,
            num_train_epochs=self.train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            eval_strategy='epoch',
            save_strategy='epoch',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='chrfpp',
            save_total_limit=3,
            predict_with_generate=True,
            dataloader_pin_memory=torch.cuda.is_available(),
            greater_is_better=True,
            fp16=(self.dtype == torch.float16),
            bf16=(self.dtype == torch.bfloat16),
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            compute_metrics=self.compute_metrics,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )


        # start training
        print('Starting training...')

        trainer.train()

        best_model_output_path = self.train_output_path + 'best-model/'
        trainer.save_model(best_model_output_path)
        self.tokenizer.save_pretrained(best_model_output_path)
        print('Model saved to ' + best_model_output_path)

        training_losses_output_path = self.train_output_path + 'losses.csv'
        self.save_training_losses(trainer, training_losses_output_path)
        print('Training losses saved to ' + training_losses_output_path)

        