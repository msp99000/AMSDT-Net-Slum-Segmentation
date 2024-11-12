import yaml
from loguru import logger
import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import pandas as pd
from architecture.loss import CombinedLoss
from utils import get_training_augmentation, get_validation_augmentation, get_loaders
from architecture.model import SlumSegNet
from captum.attr import IntegratedGradients, visualization as viz
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

class ModelComparisonFramework:
    def __init__(self, config_path, device="cpu"):
        
        # Initialize the framework with configuration and device
        self.device = device
        
        # init config and logger
        self._load_config(config_path)

        # setup directories
        self._setup_directories()

        # setup logging
        self.logger = self._setup_logging()
        
        # init the models to train & compare
        self.models = self._initialize_models()

        # init loss for training
        self.loss = CombinedLoss(self.config["loss"]["alpha"])
        
        # init metrics to calculate
        self.metrics = self._setup_metrics()
        
        # init data loaders
        self.train_loader, self.valid_loader = get_loaders(
            self.image_dir, self.mask_dir, 
            self.config["train_params"]["batch_size"], 
            get_training_augmentation(), get_validation_augmentation()
        )
        self.history = {}

    def _load_config(self, config_path):

        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def _setup_directories(self):

        # Set up necessary directories for logs, models, and plots
        self.image_dir = Path(self.config['dataset']['image_dir'])
        self.mask_dir = Path(self.config['dataset']['mask_dir'])
        self.working_dir = Path(self.config['train_params']['log_dir'])
        self.model_dir = self.working_dir / "models"
        self.plots_dir = self.working_dir / "plots"
        self.logs_dir = self.working_dir / "logs"

        for dir in [self.working_dir, self.model_dir, self.plots_dir, self.logs_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        # Configure logging with Loguru
        logger.add(self.logs_dir / "full_logs.log", 
                          colorize=True, 
                          format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> <level>{message}</level>", 
                          level="INFO")
        return logger

    def _initialize_models(self):

        # Initialize SlumSegNet and baseline models
        models = {
            "SlumSegNet": SlumSegNet(
                self.config['model_params']
                # self.config['model_params']['num_classes']
            ).to(self.device)
        }

        for name, params in self.config['model_params']['baseline_models'].items():
            models[name] = getattr(smp, name)(
                encoder_name=params['encoder'],
                encoder_weights=params['weights'],
                in_channels=self.config['model_params']['input_channels'],
                classes=self.config['model_params']['num_classes'],
            ).to(self.device)

        return models

    from sklearn.metrics import precision_score, recall_score

    def _setup_metrics(self):
        # Define your evaluation metrics here, including precision and recall
        metrics = {
            "accuracy": self.compute_accuracy,
            "f1_score": self.compute_f1_score,
            "iou_score": self.compute_iou,
            "precision": self.compute_precision,
            "recall": self.compute_recall
        }
        return metrics

    def compute_accuracy(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return accuracy_score(y_true, y_pred)

    def compute_f1_score(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred, average='binary')  # or 'macro' depending on your need

    def compute_iou(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def compute_precision(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return precision_score(y_true, y_pred, average='binary')  # or 'macro' depending on your need

    def compute_recall(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return recall_score(y_true, y_pred, average='binary')  # or 'macro' depending on your need


    def train_and_evaluate_model(self, model, model_name):

        # Train and evaluate a single model
        train_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=self.config['train_params']['learning_rate']
        )
        scheduler = ReduceLROnPlateau(train_optimizer, mode='max', factor=0.1, patience=10, verbose=True)

        trainer = smp.utils.train.TrainEpoch(
            model, loss=self.loss, metrics=self.metrics, 
            optimizer=train_optimizer, device=self.device, verbose=True,
        )
        validator = smp.utils.train.ValidEpoch(
            model, loss=self.loss, metrics=self.metrics, 
            device=self.device, verbose=True,
        )

        best_iou = 0
        early_stop_counter = 0
        history = []

        for epoch in range(self.config['train_params']['epochs']):
            self.logger.info(f'Epoch: {epoch + 1}')
            train_logs = trainer.run(self.train_loader)
            valid_logs = validator.run(self.valid_loader)
            
            scheduler.step(valid_logs['iou_score'])
            
            history.append({
                'epoch': epoch + 1,
                'train': train_logs,
                'valid': valid_logs
            })

            if valid_logs['iou_score'] > best_iou:
                best_iou = valid_logs['iou_score']
                torch.save(model.state_dict(), self.model_dir / f"best_model_{model_name}.pth")
                self.logger.info(f"New best model saved with IOU: {best_iou}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                self.logger.info(f"No improvement. Early stop counter: {early_stop_counter}/{self.config['train_params']['early_stopping_patience']}")

                if early_stop_counter >= self.config['train_params']['early_stopping_patience']:
                    self.logger.info("Early stopping triggered.")
                    break

        self.history[model_name] = history
        return history

    def conduct_ablation_study(self, model_name, model):

        # Perform ablation study by disabling model components
        self.logger.info(f"Performing ablation study for {model_name}")
        
        ablation_result = {"Full Model": self.history[model_name]}
        
        components = {
            'multi_scale': 'Multi-Scale Feature Extraction',
            'transformer_encoder': 'Transformer Encoder',
            'boundary_refinement': 'Boundary Refinement'
        }
        
        for component, description in components.items():
            if hasattr(model, component):
                self.logger.info(f"Ablation Study: Disabling {description}")
                setattr(model, component, nn.Identity())
                history = self.train_and_evaluate_model(model, f"{model_name}_no_{component}")
                ablation_result[f'No {description}'] = history
        
        self._plot_ablation_results(ablation_result, model_name)

    def _plot_ablation_results(self, ablation_result, model_name):

        # Plot results of the ablation study
        metrics = ['iou_score', 'f1_score', 'accuracy', 'recall', 'precision']
        ablation_results = {metric: [] for metric in metrics}
        ablation_types = []

        for ablation_type, history in ablation_result.items():
            ablation_types.append(ablation_type)
            for metric in metrics:
                ablation_results[metric].append(history[-1]['valid'][metric])

        df_ablation = pd.DataFrame(ablation_results, index=ablation_types)
        df_ablation.plot(kind='bar', figsize=(12, 8))
        plt.title(f'Ablation Study - {model_name}', fontsize=16)
        plt.ylabel('Score', fontsize=14)
        plt.xlabel('Ablation Type', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{model_name}_ablation_study_comparison.png')
        plt.close()

    def conduct_feature_study(self, model, model_name):

        # Perform feature importance study using Integrated Gradients
        self.logger.info(f"Performing feature study for {model_name}")
        model.load_state_dict(torch.load(self.model_dir / f"best_model_{model_name}.pth"))
        model.eval()

        sample_image, sample_mask = next(iter(self.valid_loader))
        sample_image = sample_image.to(self.device)
        sample_mask = sample_mask.to(self.device)

        ig = IntegratedGradients(model)
        attr = ig.attribute(sample_image, target=1, n_steps=self.config["kwargs"]["n_steps"])

        attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
        sample_image_np = np.transpose(sample_image.squeeze().cpu().numpy(), (1, 2, 0))
        sample_mask_np = sample_mask.squeeze().cpu().numpy()

        self._plot_feature_importance(sample_image_np, sample_mask_np, attr_np, model_name)

    def _plot_feature_importance(self, sample_image_np, sample_mask_np, attr_np, model_name):

        # Plot feature importance results
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        axs[0].imshow(sample_image_np)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(sample_mask_np, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[1].axis('off')

        axs[2].imshow(attr_np, cmap='hot')
        axs[2].set_title(f"Feature Importance (Raw) - {model_name}")
        axs[2].axis('off')

        viz.visualize_image_attr(
            attr_np,
            sample_image_np,
            method='heat_map',
            sign='positive',
            show_colorbar=True,
            plt_fig_axis=(fig, axs[3]),
            title=f"Overlayed Heatmap - {model_name}"
        )

        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{model_name}_feature_importance.png')
        plt.close()

    def visualize_training_results(self):

        # Visualize training results for all models
        metrics = ['iou_score', 'f1_score', 'accuracy', 'recall', 'precision']
        epochs = range(1, len(next(iter(self.history.values()))) + 1)

        data = {metric: {model: [h['valid'][metric] for h in history] 
                         for model, history in self.history.items()} 
                for metric in metrics}

        df = pd.DataFrame(data)

        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))

        for i, metric in enumerate(metrics):
            for model_name in df[metric]:
                axs[i].plot(epochs, df[metric][model_name], label=f'{model_name} - Validation')
            
            axs[i].set_title(f'{metric.capitalize()} over Epochs', fontsize=14)
            axs[i].set_xlabel('Epochs', fontsize=12)
            axs[i].set_ylabel(metric.capitalize(), fontsize=12)
            axs[i].legend(loc='best')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png')
        plt.close()

        final_metrics = {model: {metric: history[-1]['valid'][metric] 
                                 for metric in metrics} 
                         for model, history in self.history.items()}
        final_metrics_df = pd.DataFrame(final_metrics).T

        final_metrics_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Final Metrics Comparison', fontsize=16)
        plt.ylabel('Score', fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'final_metrics_comparison_barchart.png')
        plt.close()

    def compare_models(self):

        # Main method to compare all models
        for model_name, model in self.models.items():
            self.logger.info(f"\nTraining and evaluating {model_name}")
            self.train_and_evaluate_model(model, model_name)
            self.conduct_feature_study(model, model_name)
    
        self.visualize_training_results()

        first_model_name, first_model = next(iter(self.models.items()))
        self.conduct_ablation_study(first_model_name, first_model)

    def k_fold_cross_validation(self, k=5):

        # Perform k-fold cross-validation
        dataset = torch.utils.data.ConcatDataset([self.train_loader.dataset, self.valid_loader.dataset])
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            fold_results = []
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                self.logger.info(f'FOLD {fold}')
                self.logger.info('--------------------------------')

                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

                trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=self.config["train_params"]["batch_size"], sampler=train_subsampler)
                valloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config["train_params"]["batch_size"], sampler=val_subsampler)

                model.apply(self._weight_reset)
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train_params']['learning_rate'])
                
                for epoch in range(self.config['train_params']['epochs']):
                    # Train the model
                    model.train()
                    for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{self.config['train_params']['epochs']}"):
                        images, masks = batch
                        images, masks = images.to(self.device), masks.to(self.device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = self.loss(outputs, masks)
                        loss.backward()
                        optimizer.step()

                # Evaluate the model
                model.eval()
                fold_iou = smp.utils.metrics.IoU(threshold=0.5)
                with torch.no_grad():
                    for batch in valloader:
                        images, masks = batch
                        images, masks = images.to(self.device), masks.to(self.device)
                        outputs = model(images)
                        fold_iou.update(outputs, masks)
                
                fold_results.append(fold_iou.compute())
                
            self.logger.info(f'{model_name} - Mean IoU: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})')

    @staticmethod
    def _weight_reset(m):
        # Reset the weights of convolutional and linear layers
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def run_full_comparison(self):

        # Perform a full comparison of all models
        self.logger.info("Starting full model comparison")
        self.compare_models()
        self.logger.info("Performing k-fold cross-validation")
        self.k_fold_cross_validation(k=3)
        self.logger.info("Model comparison completed")

if __name__ == "__main__":
    # Set up the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create comparison framework
    comparison = ModelComparisonFramework("config.yaml", device=device)
    
    # Run full comparison
    comparison.run_full_comparison()
