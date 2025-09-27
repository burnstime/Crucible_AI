from celery import Celery
import os
import time
import logging

celery_app = Celery(
    "crucible", broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
)


@celery_app.task
def run_retrain_job():
    log_path = f"experiments/run_logs/{int(time.time())}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO)
    try:
        from crucible.tools.data_cleaning import clean_logs
        from crucible.trainer.train import main as train_main
        from crucible.tests.test_safety import run_safety_suite

        clean_logs()
        train_main()
        passed = run_safety_suite()
        if passed:
            from crucible.tools.storage import promote_candidate

            promote_candidate("candidate", promote=True)
        logging.info("Retrain job completed successfully.")
        return 0
    except Exception as e:
        logging.error(f"Retrain job failed: {e}")
        return 1
