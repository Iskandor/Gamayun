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
        self.templates = {
            'a2': Template(),
            'snd': Template(),
            'seer': Template(),
        }

        self.templates['snd'].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates['snd'].add_element('score', ['sum'], 'blue', 'score')
        self.templates['snd'].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates['snd'].add_element('feature_space', ['mean', 'std'], 'green', 'feature space L2 norm')
        self.templates['snd'].add_element('loss_prediction', ['val'], 'magenta', 'loss prediction')
        self.templates['snd'].add_element('loss_target', ['val'], 'magenta', 'loss target')

        self.templates['a2'].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates['a2'].add_element('score', ['sum'], 'blue', 'score')
        self.templates['a2'].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates['a2'].add_element('feature_a_norm', ['mean', 'std'], 'green', 'feature space A L2 norm')
        self.templates['a2'].add_element('feature_b_norm', ['mean', 'std'], 'green', 'feature space B L2 norm')
        self.templates['a2'].add_element('loss_encoder', ['val'], 'magenta', 'encoder loss')
        self.templates['a2'].add_element('loss_hidden', ['val'], 'magenta', 'hidden loss')
        self.templates['a2'].add_element('loss_associative', ['val'], 'magenta', 'associative loss')

        self.templates['seer'].add_element('re', ['sum'], 'blue', 'external reward')
        self.templates['seer'].add_element('score', ['sum'], 'blue', 'score')
        self.templates['seer'].add_element('ri', ['mean', 'std'], 'red', 'intrinsic reward')
        self.templates['seer'].add_composite([('distillation_reward', 'red', 'distillation reward'),
                                              ('forward_reward', 'darkred', 'forward reward')],
                                             ['mean', 'std'])
        self.templates['seer'].add_composite([('target_space', 'green', 'target space L2 norm'),
                                              ('learned_space', 'darkgreen', 'learned space L2 norm'),
                                              ('forward_space', 'limegreen', 'forward space L2 norm')],
                                             ['mean', 'std'])
        # self.templates['seer'].add_element('next_space', ['mean', 'std'], 'green', 'next space L2 norm')
        self.templates['seer'].add_element('hidden_space', ['mean', 'std'], 'green', 'hidden space L2 norm')
        self.templates['seer'].add_element('confidence', ['mean', 'std'], 'orange', 'confidence')
        self.templates['seer'].add_element('loss_target', ['val'], 'magenta', 'encoder loss')
        self.templates['seer'].add_element('loss_distillation', ['val'], 'magenta', 'distillation loss')
        self.templates['seer'].add_element('loss_forward', ['val'], 'magenta', 'forward model loss')
        self.templates['seer'].add_element('loss_hidden', ['val'], 'magenta', 'hidden loss')

    def __getitem__(self, key):
        return self.templates[key]
