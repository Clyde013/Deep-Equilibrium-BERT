from transformers.models.roberta.configuration_roberta import RobertaConfig


class DEQBertConfig(RobertaConfig):
    model_type = "deqbert"

    def __init__(self,
                 f_solver="anderson",
                 b_solver="broyden",
                 f_thres=100,
                 b_thres=100,
                 jac_loss_freq=0.35,
                 jac_loss_weight=1.9,
                 **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(**kwargs)

        self.jac_loss_freq = jac_loss_freq
        self.jac_loss_weight = jac_loss_weight

        self.f_thres = f_thres
        self.b_thres = b_thres

        self.f_solver = f_solver
        self.b_solver = b_solver
