from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging
from src.tasks.tasks import celery_app

logger = setup_logging("src.tasks.cli")
evaluator = Evaluator()

if Evaluator().validate_llm_judge_availability():
    logger.info("Evaluator LLM judge is available!")
    checked_evaluator_llm_judge_availability = True
else:
    logger.error("""
    **************************************************
    *                                                *
    *  Remove Evaluator LLM judge is not available!  *
    *  Did you set the correct API key?              *
    *  `NGC_API_KEY` needs to be set.                *
    *                                                *
    *  Exiting                                       *
    *                                                *
    **************************************************
    """)
    exit(1)


evaluator.validate_llm_judge_availability()
logger.info(f"Loaded {celery_app}")
