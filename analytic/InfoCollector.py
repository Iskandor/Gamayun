class InfoPoint:
    def __init__(self, subkey, label, priority=0):
        self.subkey = subkey
        self.priority = priority
        self.label = label

    def __gt__(self, ip):
        return self.priority > ip.priority


class InfoCollector:
    def __init__(self, trial, step_counter, reward_average, info_points=None):
        self.trial = trial
        self.step_counter = step_counter
        self.reward_average = reward_average
        self.info_points = {}

        for ip in info_points:
            self.info_points[ip[0]] = InfoPoint(ip[1], ip[2], ip[3])
        self._sort()

    def add_info_point(self, key, subkey, label, priority=0):
        self.info_points[key] = InfoPoint(subkey, label, priority)
        self._sort()

    def _sort(self):
        self.info_points = dict(sorted(self.info_points.items(), key=lambda x: x[1], reverse=True))

    def print(self, data, index):
        result = 'Run {0:d} step {1:d}/{2:d} average reward {3:f} | '.format(self.trial, self.step_counter.steps, self.step_counter.limit, self.reward_average.value().item())
        for key in self.info_points:
            root_value = data[key]
            result += '{0:s}: '.format(self.info_points[key].label)
            for subkey in self.info_points[key].subkey:
                value = getattr(root_value, subkey)[index]

                if subkey == 'max':
                    result += 'max {0:.4f} '.format(value)
                elif subkey == 'mean':
                    result += '\u03BC {0:.4f} '.format(value)
                elif subkey == 'std':
                    result += '\u00B1 {0:.4f} '.format(value)
                elif subkey == 'step':
                    result += '| {0} steps '.format(int(value))
                else:
                    result += '{0} '.format(value)

            result += '| '

        print(result[:-1])
