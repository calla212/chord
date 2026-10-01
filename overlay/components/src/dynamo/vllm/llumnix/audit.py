from dataclasses import asdict
import json
import logging
import time

logger = logging.getLogger(__name__)


def migration_audit(event, decision, **extra):
    if decision is None or decision.policy.value != "FCWSR":
        return
    value = asdict(decision)
    value.update(event=event, wall_time=time.time(), **extra)
    value["policy"] = decision.policy.value
    logger.info("FCWSR_AUDIT %s", json.dumps(value, sort_keys=True))
