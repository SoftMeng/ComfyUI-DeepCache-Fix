try:
	import comfy.utils
except ImportError:
	pass
else:
	from .DeepCache_Fix import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
	__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
