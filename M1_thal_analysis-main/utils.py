class MetaParams(dict):

    def __init__(self, base, params_dict):
        super().__init__(params_dict)
        self.base = base

    def full_label(self):
        # generate label like "base_one0_two1_three2" where one two three are param short names, 0 1 and 2 are values
        label = self.base
        for key in self.keys():
            value = self.val(key)
            incl_name = self.include_param_name(key)
            if not incl_name:
                key = ''
            if isinstance(value, bool):
                if value:
                    label += f"_{key}"
            else:
                label += f"_{key}{value}"
        return label

    def val(self, param, default=None):
        if param not in self:
            return default
        return self.get(param)[0]

    def description(self, param, default=None):
        if param not in self:
            return None
        return self.get(param)[1]
    
    def include_param_name(self, param, default=None):
        if param not in self:
            return None
        all = self.get(param)
        if len(all) < 3:
            return True
        else:
            return all[2]

