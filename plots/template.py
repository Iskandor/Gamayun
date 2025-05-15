class TemplateElement:
    def __init__(self, key, values, color, legend):
        self.key = key
        self.values = values
        self.color = color
        self.legend = legend


class Template:
    def __init__(self):
        self.elements = []

    def add_element(self, key, values, color, legend):
        self.elements.append([TemplateElement(key, values, color, legend)])

    def add_composite(self, description, values):
        composite = []
        for key, color, legend in description:
            composite.append(TemplateElement(key, values, color, legend))
        self.elements.append(composite)


class ChartTemplates:
    def __init__(self):
        A2 = 'a2'
        DPM = 'dpm'
        SND = 'snd'
        SNDV2 = 'sndv2'
        SEER = 'seer'
        FWD = 'fm'
        IJEPA = 'ijepa'

        self.templates = {
            A2: Template(),
            DPM: Template(),
            SND: Template(),
            SNDV2: Template(),
            SEER: Template(),
            FWD: Template(),
            IJEPA: Template()
        }

        self.templates[SND].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[SND].add_element('score', ['sum'], 'blue', 'score')
        self.templates[SND].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates[SND].add_element('feature_space', ['mean', 'std'], 'green', 'feature space L2 norm')
        self.templates[SND].add_element('loss_prediction', ['val'], 'magenta', 'loss prediction')
        self.templates[SND].add_element('loss_target', ['val'], 'magenta', 'loss target')

        self.templates[FWD].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[FWD].add_element('score', ['sum'], 'blue', 'score')
        self.templates[FWD].add_element('ri', ['mean', 'std', 'max'], 'red', 'intrinsic reward')
        self.templates[FWD].add_element('feature_space', ['mean', 'std'], 'green', 'feature space L2 norm')
        self.templates[FWD].add_element('error', ['mean', 'std', 'max'], 'green', 'error')

        self.templates[IJEPA].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[IJEPA].add_element('score', ['sum'], 'blue', 'score')
        self.templates[IJEPA].add_element('ri', ['mean', 'std', 'max'], 'red', 'intrinsic reward')
        self.templates[IJEPA].add_element('feature_space', ['mean', 'std'], 'green', 'feature space L2 norm')
        self.templates[IJEPA].add_element('error', ['mean', 'std', 'max'], 'green', 'error')

        self.templates[SNDV2].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[SNDV2].add_element('score', ['sum'], 'blue', 'score')
        self.templates[SNDV2].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates[SNDV2].add_element('feature_space', ['mean', 'std'], 'green', 'feature space L2 norm')
        self.templates[SNDV2].add_element('loss_prediction', ['val'], 'magenta', 'loss prediction')
        self.templates[SNDV2].add_element('loss_target', ['val'], 'magenta', 'loss target')

        self.templates[DPM].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[DPM].add_element('score', ['sum'], 'blue', 'score')
        self.templates[DPM].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates[DPM].add_element('ppo_space', ['mean', 'std'], 'green', 'PPO space')
        self.templates[DPM].add_element('target_space', ['mean', 'std'], 'green', 'SND target space')
        self.templates[DPM].add_element('forward_space', ['mean', 'std'], 'green', 'forward space')
        self.templates[DPM].add_composite([('loss_prediction_t0', 'magenta', 'loss prediction t0'),
                                           ('loss_prediction_tH', 'orchid', 'loss prediction tH')], ['val'])
        self.templates[DPM].add_element('loss_distillation', ['val'], 'magenta', 'loss distillation')
        self.templates[DPM].add_element('loss_target', ['val'], 'magenta', 'loss target')
        self.templates[DPM].add_element('distillation_error', ['mean', 'std'], 'red', 'distillation error')
        self.templates[DPM].add_element('prediction_error', ['mean', 'std'], 'red', 'prediction error')

        self.templates[A2].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[A2].add_element('score', ['sum'], 'blue', 'score')
        self.templates[A2].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates[A2].add_composite([('za_state_norm', 'darkgreen', 'zA space L2 norm'),
                                          ('zb_state_norm', 'limegreen', 'zB space L2 norm')], ['mean', 'std'])
        self.templates[A2].add_element('loss_encoder', ['val'], 'magenta', 'encoder loss')
        self.templates[A2].add_element('loss_hidden', ['val'], 'magenta', 'hidden loss')
        self.templates[A2].add_element('loss_associative', ['val'], 'magenta', 'associative loss')

        self.templates[SEER].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates[SEER].add_element('score', ['sum'], 'blue', 'score')
        self.templates[SEER].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates[SEER].add_composite([('distillation_reward', 'red', 'distillation reward'),
                                            ('forward_reward', 'darkred', 'forward reward')],
                                           ['mean', 'std'])
        self.templates[SEER].add_composite([('target_space', 'green', 'target space L2 norm'),
                                            ('learned_space', 'darkgreen', 'learned space L2 norm'),
                                            ('forward_space', 'limegreen', 'forward space L2 norm')],
                                           ['mean', 'std'])
        # self.templates[SEER].add_element('next_space', ['mean', 'std'], 'green', 'next space L2 norm')
        self.templates[SEER].add_element('hidden_space', ['mean', 'std'], 'green', 'hidden space L2 norm')
        self.templates[SEER].add_element('confidence', ['mean', 'std'], 'orange', 'confidence')
        self.templates[SEER].add_element('loss_target', ['val'], 'magenta', 'encoder loss')
        self.templates[SEER].add_element('loss_distillation', ['val'], 'magenta', 'distillation loss')
        self.templates[SEER].add_element('loss_forward', ['val'], 'magenta', 'forward model loss')
        self.templates[SEER].add_element('loss_hidden', ['val'], 'magenta', 'hidden loss')

    def __getitem__(self, key):
        return self.templates[key]
