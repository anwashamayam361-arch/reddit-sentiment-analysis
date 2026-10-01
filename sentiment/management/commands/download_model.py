from django.conf import settings
from django.core.management.base import BaseCommand
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Command(BaseCommand):
    help = "Download the Hugging Face sentiment model into ml_models/ (run once)."

    def handle(self, *args, **options):
        model_name = settings.HF_MODEL_NAME
        save_dir = settings.BASE_DIR / "ml_models" / "sentiment"
        save_dir.mkdir(parents=True, exist_ok=True)

        self.stdout.write(f"Downloading '{model_name}' ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        self.stdout.write(self.style.SUCCESS(f"Model saved to {save_dir}"))