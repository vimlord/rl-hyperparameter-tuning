
def train(module, args):
    agent = module(**args)
    return agent.cpu().state_dict()

