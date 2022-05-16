import torch
import copy

def model_sum(params1, params2):
	with torch.no_grad():
		for k in params1.keys():
			params1[k] += params2[k]

def model_avg(param_sum, param_count, args, origin):
	w_avg = copy.deepcopy(param_sum)

	with torch.no_grad():
		for k in w_avg.keys():
			if param_count[k] == 0:
				w_avg[k] = origin[k]
				continue
			
			w_avg[k] = torch.div(w_avg[k], param_count[k])

			# if args.FedDyn == 1:
			# 	h[k] = h[k] - args.alpha * param_count[k] * (w_avg[k] - \
			# 		origin[k]) / args.nodes

			# 	if 'weight' not in k and 'bias' not in k:
			# 		continue
				
			# 	w_avg[k] = w_avg[k] - h[k] / args.alpha

	return w_avg


		