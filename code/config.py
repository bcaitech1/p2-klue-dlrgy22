from dataclasses import dataclass

@dataclass
class Config:
    device: str
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    MODEL_NAME:str
    ADD_VOCAB: bool
    ADD_LSTM: bool
    MASK: bool
    SWITCH: bool