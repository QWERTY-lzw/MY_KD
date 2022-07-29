from mmrazor.models.builder import PRUNERS, build_mutable
from mmrazor.models.pruners import RatioPruner
from collections import OrderedDict
from mmrazor.models.pruners.utils import SwitchableBatchNorm2d
import torch
import torch.nn as nn


@PRUNERS.register_module()
class DetRatioPrunerDev(RatioPruner):
    def sample_subnet(self):
        pass

    def set_subnet(self, subnet_dict):
        pass
    
    def build_channel_spaces(self, name2module):
        """Build channel search space.

        Args:
            name2module (dict): A mapping between module_name and module.

        Return:
            dict: The channel search space. The key is space_id and the value
                is the corresponding out_mask.
        """
        search_space = dict()

        for module_name in self.modules_have_child:
            need_prune = True
            for key in self.except_start_keys:
                if module_name.startswith(key):
                    need_prune = False
                    break
            if not need_prune:
                continue
            if module_name in self.module2group:
                space_id = self.module2group[module_name]
            else:
                space_id = module_name
            module = name2module[module_name]
            if space_id not in search_space:
                search_space[space_id] = module.out_mask

        return search_space

    # supernet is a kind of architecture in `mmrazor/models/architectures/`
    def prepare_from_supernet(self, supernet):
        """Prepare for pruning."""

        module2name = OrderedDict()
        name2module = OrderedDict()
        var2module = OrderedDict()

        # record the visited module name during trace path
        visited = dict()
        # Record shared modules which will be visited more than once during
        # forward such as shared detection head in RetinaNet.
        # If a module is not a shared module and it has been visited during
        # forward, its parent modules must have been traced already.
        # However, a shared module will be visited more than once during
        # forward, so it is still need to be traced even if it has been
        # visited.
        self.shared_module = []
        tmp_shared_module_hook_handles = list()

        for name, module in supernet.model.named_modules():
            if hasattr(module, 'weight'):
                module2name[module] = name
                name2module[name] = module
                var2module[id(module.weight)] = module
                self.add_pruning_attrs(module)
            if isinstance(module, SwitchableBatchNorm2d):
                name2module[name] = module
        self.name2module = name2module
        self.module2name = module2name

        # Set requires_grad to True. If the `requires_grad` of a module's
        # weight is False, we can not trace this module by parsing backward.
        param_require_grad = dict()
        for param in supernet.model.parameters():
            param_require_grad[id(param)] = param.requires_grad
            param.requires_grad = True

        self.search_spaces = self.build_search_spaces(supernet)

        self._reset_norm_running_stats(supernet)


@PRUNERS.register_module()
class DetRatioPruner(RatioPruner):
    # def set_subnet(self, subnet_dict):
    #     """Modify the in_mask and out_mask of modules in supernet according to
    #     subnet_dict.

    #     Args:
    #         subnet_dict (dict): the key is space_id and the value is the
    #             corresponding sampled out_mask.
    #     """
    #     def add_out_mask(module, space_id):
    #         if space_id in subnet_dict:
    #             module.out_mask = subnet_dict[space_id].to(module.out_mask.device)
    #     def add_in_mask(module, space_id):
    #         if space_id in subnet_dict:
    #             module.in_mask = subnet_dict[space_id].to(module.in_mask.device)

    #     for module_name in self.modules_have_child:
    #         space_id = self.get_space_id(module_name)
    #         module = self.name2module[module_name]
    #         # module.out_mask = subnet_dict[space_id].to(module.out_mask.device)
    #         add_out_mask(module, space_id)

    #     for bn, conv in self.bn_conv_links.items():
    #         module = self.name2module[bn]
    #         conv_space_id = self.get_space_id(conv)
    #         # conv_space_id is None means the conv layer in front of
    #         # this bn module can not be pruned. So we should not set
    #         # the out_mask of this bn layer
    #         # if conv_space_id is not None:
    #         #     module.out_mask = subnet_dict[conv_space_id].to(
    #         #         module.out_mask.device)
    #         add_out_mask(module, conv_space_id)

    #     for module_name in self.modules_have_ancest:
    #         need_prune = True
    #         for key in self.except_start_keys:
    #             if module_name.startswith(key):
    #                 need_prune = False
    #                 break
    #         if not need_prune:
    #             continue
    #         module = self.name2module[module_name]
    #         parents = self.node2parents[module_name]
    #         # To avoid ambiguity, we only allow the following two cases:
    #         # 1. all elements in parents are ``Conv2d``,
    #         # 2. there is only one element in parents, ``concat`` or ``chunk``
    #         # In case 1, all the ``Conv2d`` share the same space_id and
    #         # out_mask.
    #         # So in all cases, we only need the very first element in parents
    #         parent = parents[0]
    #         space_id = self.get_space_id(parent)

    #         if isinstance(space_id, dict):
    #             if 'concat' in space_id:
    #                 in_mask = []
    #                 for parent_space_id in space_id['concat']:
    #                     in_mask.append(subnet_dict[parent_space_id])
    #                 module.in_mask = torch.cat(
    #                     in_mask, dim=1).to(module.in_mask.device)
    #         else:
    #             # module.in_mask = subnet_dict[space_id].to(
    #             #     module.in_mask.device)
    #             add_in_mask(module, space_id)


    def set_subnet(self, subnet_dict):
        """Modify the in_mask and out_mask of modules in supernet according to
        subnet_dict.

        Args:
            subnet_dict (dict): the key is space_id and the value is the
                corresponding sampled out_mask.
        """
        for module_name in self.modules_have_child:
            space_id = self.get_space_id(module_name)
            module = self.name2module[module_name]
            module.out_mask = subnet_dict[space_id].to(module.out_mask.device)

        for bn, conv in self.bn_conv_links.items():
            module = self.name2module[bn]
            conv_space_id = self.get_space_id(conv)
            # conv_space_id is None means the conv layer in front of
            # this bn module can not be pruned. So we should not set
            # the out_mask of this bn layer
            if conv_space_id is not None:
                module.out_mask = subnet_dict[conv_space_id].to(
                    module.out_mask.device)


        def concat_in_mask(space_id):
            if 'concat' in space_id:
                in_mask = []
                for parent_space_id in space_id['concat']:
                    in_mask.append(concat_in_mask(parent_space_id))
                return torch.cat(in_mask, dim=1)
            else:
                return subnet_dict[space_id]

        for module_name in self.modules_have_ancest:
            module = self.name2module[module_name]
            parents = self.node2parents[module_name]
            # To avoid ambiguity, we only allow the following two cases:
            # 1. all elements in parents are ``Conv2d``,
            # 2. there is only one element in parents, ``concat`` or ``chunk``
            # In case 1, all the ``Conv2d`` share the same space_id and
            # out_mask.
            # So in all cases, we only need the very first element in parents
            parent = parents[0]
            space_id = self.get_space_id(parent)

            if isinstance(space_id, dict):
                if 'concat' in space_id:
                    # in_mask = []
                    # for parent_space_id in space_id['concat']:
                    #     in_mask.append(subnet_dict[parent_space_id])
                    # module.in_mask = torch.cat(
                    #     in_mask, dim=1).to(module.in_mask.device)
                    in_mask = concat_in_mask(space_id)
                    module.in_mask = in_mask.to(module.in_mask.device)
            else:
                module.in_mask = subnet_dict[space_id].to(
                    module.in_mask.device)