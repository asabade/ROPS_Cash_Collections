
class Inference(metaclass=ABCMeta):
	"""docstring for Inference"""
	def __init__(self, datapackager, model):
		super(Inference, self).__init__()
		#self.datapackager = datapackager
		self.model = model
	
    @classmethod
    def from_checkpoint(cls, datapackager, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        arch_prefix = checkpoint['architecture']['prefix']
        arch_suffix = checkpoint['architecture']['name']
        build_params = checkpoint['architecture']['hyperparameters']

        arch = getattr(arch_prefix, arch_suffix).inference(build_params)
        model = arch.model
        return cls(datapackager, architecture)

    @classmethod
    def from_source_and_checkpoint(cls, query, config_dir, checkpoint_path):
    	datapackager = DataPackager.from_query(query=query, config_dir=config_dir)
    	model = checkpoint_to_model(checkpoint_path)
    	return cls(datapackager=datapackager, model=model) 

	@abstractmethod
	def predict(self, datapackager):
		retval = []
		for batch_idx, batch_dict in enumerate(datapackager['inference']):
			yhat = self.model(cat_X=batch_dict['cat_X'], cont_X=batch_dict['cont_X'])
			retval.append(yhat)

		return retval