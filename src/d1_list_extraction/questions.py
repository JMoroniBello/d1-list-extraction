"""Exact question profile used in the original experiments; order is intentional."""
KINDS = ("people", "products", "organizations", "combined")
QUESTIONS = {
    "people": {"type": "text", "instructions": "Find an explicitly named individual person in the text. Return just one person-name mention. Exclude organizations, products, pronouns and unnamed roles. Return null if no individual person is named."},
    "products": {"type": "text", "instructions": "Find a software product or platform explicitly named in the text. Return just one product-name mention. Exclude service providers, organizations, people and generic unnamed products. Return null if no software product is named."},
    "organizations": {"type": "text", "instructions": "Find an explicitly named organization in the text. Return just one organization-name mention. Exclude individual people, software products and unnamed organizations. Return null if no organization is named."},
    "combined": {"type": "text", "instructions": "Find an explicitly named person, organization or software product in the text. Return just one name mention, not a phrase containing several names. Exclude pronouns, unnamed roles, generic nouns and dates. Return null if none of these named entities are present."},
}

