import os
import argparse

import numpy as np
import cv2 as cv2
import pandas as pd
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import wandb

from src.dataset import CustomDataModule
from src.model import create_model


SEED = 36
L.seed_everything(SEED)
class ClassficationModel(L.LightningModule):
    def __init__(self, model, batch_size: int = 32, learning_rate: float = 0.0001, dataset_type: str = 'flowers'):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.losses = []
        self.labels = []
        self.predictions = []
        self.automatic_optimization = False
        self.dataset_type = dataset_type  # 데이터셋 타입 저장
        self.num_classes = 10 if dataset_type == 'cifar10' else 5  

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        if self.model.classifier_type != 'cnn':
            self.model.set_targets(target)
            output = self.model(inputs)
            output = torch.nn.functional.one_hot(output.long(), num_classes=self.num_classes).float()
            loss = self.loss_fn(output, target)
            self.log('train_loss', loss)
            return loss
        else:
            opt = self.optimizers()
            opt.zero_grad()
            output = self.model(inputs)
            loss = self.loss_fn(output, target)
            self.manual_backward(loss)
            opt.step()
            self.log('train_loss', loss)
            return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        if self.model.classifier_type != 'cnn':
            self.model.set_targets(target)
        output = self.model(inputs)
        
        # sklearn 분류기의 경우 출력을 one-hot 인코딩으로 변환
        if self.model.classifier_type != 'cnn':
            output = torch.nn.functional.one_hot(output.long(), num_classes=self.num_classes).float()
        
        loss = self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('valid_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype=object))
        predictions = np.concatenate(np.array(self.predictions, dtype=object))
        acc = sum(labels == predictions)/len(labels)

        labels = labels.tolist()
        predictions = predictions.tolist()
        loss = sum(self.losses)/len(self.losses)

        self.log('val_epoch_acc', acc)
        self.log('val_epoch_loss', loss)
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()
    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        if self.model.classifier_type != 'cnn':
            self.model.set_targets(target)
        output = self.model(inputs)
        
        # sklearn 분류기의 경우 출력을 one-hot 인코딩으로 변환
        if self.model.classifier_type != 'cnn':
            output = torch.nn.functional.one_hot(output.long(), num_classes=self.num_classes).float()
            
        loss = self.loss_fn(output, target)
        _, predictions = torch.max(output, 1)

        target_np = target.detach().cpu().numpy()
        predict_np = predictions.detach().cpu().numpy()
        
        self.losses.append(loss)
        self.labels.append(np.int16(target_np))
        self.predictions.append(np.int16(predict_np))
        self.log('test_loss', loss)
        return loss
    
    def on_test_epoch_end(self):
        labels = np.concatenate(np.array(self.labels, dtype=object))
        predictions = np.concatenate(np.array(self.predictions, dtype=object))
        acc = sum(labels == predictions)/len(labels)

        # 혼동 행렬 계산
        cm = confusion_matrix(labels, predictions)
        
        # 클래스 이름 설정
        if self.dataset_type == 'cifar10':
            class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
                         "Dog", "Frog", "Horse", "Ship", "Truck"]
        else:  # flowers
            class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        # WandB에 혼동 행렬 이미지 로깅
        self.logger.experiment.log({
            "confusion_matrix": wandb.Image(plt),
            "test_epoch_acc": acc,
            "test_epoch_loss": sum(self.losses)/len(self.losses)
        })
        
        # 상세 분류 리포트 출력
        print("\nClassification Report:")
        print(classification_report(labels, predictions, target_names=class_names))
        
        plt.close()
        
        self.losses.clear()
        self.labels.clear()
        self.predictions.clear()

    def predict_step(self, batch, batch_idx):
        inputs, img = batch
        output = self.model(inputs)
        _, pred_cls = torch.max(output, 1)

        return pred_cls.detach().cpu().numpy(), img

    def configure_optimizers(self):
        if self.model.classifier_type == 'cnn':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_epoch_acc"
                }
            }
        else:
            # sklearn 분류기의 경우 optimizer가 필요 없음
            return None
    

def main(classification_model: str, dataset_type: str, data_path: str, batch: int,
         epoch: int, save_path: str, device: str, gpus: list, precision: str, mode: str, ckpt: str):
    
    wandb_logger = WandbLogger(project='classification')
    
    model = ClassficationModel(
        model=create_model(classification_model, dataset_type),
        batch_size=batch,
        dataset_type=dataset_type
    )
    
    if mode == 'train':
        callbacks = [
            ModelCheckpoint(
                dirpath=save_path,
                filename=f'{classification_model}-{dataset_type}-'+'{epoch:02d}-{val_epoch_acc:.3f}',
                monitor='val_epoch_acc',
                mode='max'
            ),
            EarlyStopping(
                monitor='val_epoch_acc',
                patience=5,
                mode='max'
            )
        ]
        
        # devices 파라미터를 정수로 변환
        if isinstance(gpus, list):
            num_gpus = len(gpus)
        else:
            num_gpus = 1
            
        trainer = L.Trainer(
            accelerator=device,
            devices=num_gpus,
            precision=precision,
            max_epochs=epoch,
            logger=wandb_logger,
            callbacks=callbacks
        )
        
        datamodule = CustomDataModule(dataset_type, data_path, batch)
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
    else:
        trainer = L.Trainer(
            accelerator=device,
            devices=gpus,
            precision=precision
        )
        model = ClassficationModel.load_from_checkpoint(ckpt, model=create_model(classification_model, dataset_type))
        pred_cls, img = trainer.predict(model, CustomDataModule(dataset_type, data_path, batch, mode='predict'))[0]
        
        # 데이터셋별 클래스 이름 처리
        if dataset_type == 'cifar10':
            class_names = [
                "Airplane", "Automobile", "Bird", "Cat", "Deer",
                "Dog", "Frog", "Horse", "Ship", "Truck"
            ]
            pred_label = class_names[pred_cls[0]]
        else:  # flowers
            class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]  # 예시
            pred_label = class_names[pred_cls[0]]
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (800, 600))
        cv2.putText(
            img,
            f'Predicted class: "{pred_label}"',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        cv2.imshow('Predicted output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='cnn',
                        choices=['cnn', 'knn', 'svm', 'dt', 'mlp'],
                        help='Model type to use')
    parser.add_argument('-dt', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'flowers'],
                        help='Dataset to use (cifar10 or flowers)')
    parser.add_argument('-b', '--batch_size', dest='batch', type=int, default=128)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-d', '--data_path', dest='data', type=str, default='./dataset')
    parser.add_argument('-s', '--save_path', dest='save', type=str, default='./checkpoint/')
    parser.add_argument('-dc', '--device', type=str, default='gpu')
    parser.add_argument('-g', '--gpus', type=int, default=1)
    parser.add_argument('-p', '--precision', type=str, default='32-true')
    parser.add_argument('-mo', '--mode', type=str, default='train')
    parser.add_argument('-c', '--ckpt_path', dest='ckpt', type=str, default='./checkpoint/')
    args = parser.parse_args()
    
    main(args.model, args.dataset, args.data, args.batch, args.epoch, args.save, 
         args.device, args.gpus, args.precision, args.mode, args.ckpt)