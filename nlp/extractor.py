def extract_actions(text):

    actions = []

    if "follow up" in text.lower():
        actions.append("follow up")

    return actions
