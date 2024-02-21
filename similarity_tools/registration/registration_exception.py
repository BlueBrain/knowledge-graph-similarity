# TODO define better these

class RegistrationException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SimilarityToolsException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ModelBuildingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
