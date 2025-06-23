from netspresso.enums.base import StrEnum


class ServiceTask(StrEnum):
    TRAINING = "Training"
    ADVANCED_COMPRESSION = "Advanced Compression"
    AUTOMATIC_COMPRESSION = "Automatic Compression"
    MODEL_CONVERT = "Conversion"
    MODEL_PROFILE = "Profile"
    MODEL_QUANTIZE = "Quantization"
    MODEL_GRAPH_OPTIMIZE = "Graph Optimize"
    MODEL_SIMULATE = "Simulate"


class ServiceCredit:
    CREDITS = {
        ServiceTask.ADVANCED_COMPRESSION: 50,
        ServiceTask.AUTOMATIC_COMPRESSION: 25,
        ServiceTask.MODEL_CONVERT: 50,
        ServiceTask.MODEL_PROFILE: 25,
        ServiceTask.MODEL_QUANTIZE: 50,
        ServiceTask.MODEL_GRAPH_OPTIMIZE: 50,
    }

    @staticmethod
    def get_credit(task_id):
        return ServiceCredit.CREDITS.get(task_id, "Task not found")


class MembershipType(StrEnum):
    BASIC = "BASIC"
    PRO = "PRO"
    PREMIUM = "PREMIUM"
