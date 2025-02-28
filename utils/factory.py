from methods.duct import DUCT


def get_model(model_name, args):
    name = model_name.lower()
    options = {
            'duct': DUCT,
               }
    return options[name](args)

