"""The interface a generator implements to be evaluated by MetaCF (spec section 3)."""
from __future__ import annotations


class ModelAdapter:
    name = "base"
    model_id = "base"
    checkpoint = ""
    supported_fields: set = set()     # MetaCF field names the model is conditioned on
    pathways: list = []               # optional (T9); empty = T9 is N/A
    field_groups: dict = {}           # MetaCF field name -> model field group (null masks, T9 groups)

    def generate(self, caption, fields: dict, null_mask: dict, seed: int, guidance=None):
        """-> PIL.Image. caption None = null caption. null_mask: {group: True} groups forced to the
        model's learned null."""
        raise NotImplementedError

    def generate_with_pathway(self, caption, fields, null_mask, seed, pathway, mode,
                              donor_fields=None, groups=None):
        """mode in {zero, mean, shuffle}. shuffle takes the pathway's input from donor_fields; with
        `groups`, only those field groups are taken from the donor (T9 positive control). Modes act
        on every CFG branch in which that pathway's conditioning is present; null branches are
        unchanged."""
        raise NotImplementedError(f"{self.name} exposes no pathways")

    def planted_signal(self, field: str, values):
        """Known noise-free read-back (in the field's own units) at each set value, or None. Only
        synthetic adapters know it; T5a uses it to report the theoretical MI next to the estimate."""
        return None

    def prepare_pathway_means(self, train_fields: list):
        """Called once with the fields of 4096 TRAIN records before MEAN-mode generation."""

    def generate_request(self, req, donor_fields=None):
        """GenRequest -> PIL.Image (dispatch on the variant)."""
        null = {g: True for g in req.null_mask}
        var = req.variant_dict
        if var:
            return self.generate_with_pathway(req.caption, req.field_dict, null, req.seed,
                                              var["pathway"], var["mode"], donor_fields,
                                              list(var.get("groups", ())) or None)
        return self.generate(req.caption, req.field_dict, null, req.seed, req.guidance)
