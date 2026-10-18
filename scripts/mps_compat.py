"""Keep TAL's boolean indexing on CPU while model/backpropagation run on MPS.

Applies only to this process, without modifying the installed Ultralytics package.
Reference: https://github.com/ultralytics/ultralytics/issues/22971
and the CPU-assignment workaround discussed in PR 24200.
"""


def install_safe_assignment():
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner

    if getattr(TaskAlignedAssigner, "_card_cpu_assignment", False):
        return
    original = TaskAlignedAssigner.forward

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        tensors = (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        if pd_scores.device.type != "mps":
            return original(self, *tensors)
        device = pd_scores.device
        result = original(self, *(t.detach().cpu() for t in tensors))
        return tuple(t.to(device) for t in result)

    TaskAlignedAssigner.forward = forward
    TaskAlignedAssigner._card_cpu_assignment = True
