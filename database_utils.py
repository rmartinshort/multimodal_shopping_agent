import tinydb


def get_schema_from_db(db):
    schema_options = ["tool_id", "data_type", "user_id"]
    schema = {}
    for record in db.all():
        for key, value in record.items():
            if key not in schema:
                if key in schema_options:
                    schema[key] = {"data_type": set(), "options": set()}
                else:
                    schema[key] = {"data_type": set()}
            if key in schema_options:
                schema[key]["data_type"].add(type(value))
                schema[key]["options"].add(value)
            else:
                schema[key]["data_type"].add(type(value))
    return schema


def run_generated_query(db: tinydb.TinyDB, query_string: str):
    """
    Executes a TinyDB query generated as a string.

    Args:
        db (TinyDB): The TinyDB database instance.
        query_string (str): The query string generated by the LLM.

    Returns:
        list: The result of the query.  Returns an empty list if there is an error.
    """
    try:
        # Provide the necessary context for eval: the tinydb module and the Query class
        operation = query_string.split("(")[0].split(".")[1]
        if operation not in ["search"]:
            results = []

        if operation == "search":
            query_string = query_string.replace("db.search", "")
            query = eval(query_string, {"tinydb": tinydb})
            results = db.search(query)
    except Exception as e:
        print(f"Error executing query: {e}")
        results = "Error running generated query"

    return results


def convert_database_entries_to_conversation(database_entries):
    conversation = {}
    for entry in database_entries:
        timestamp = entry["timestamp"]
        if not entry["tool_id"]:
            if entry["data_type"] == "user_input_text":
                sender = "user"
            else:
                sender = "assistant"
        else:
            if entry["data_type"] in ["input", "output"]:
                sender = "tool ({}, {})".format(entry["tool_id"], entry["data_type"])

            else:
                sender = "unknown"

        if entry["data_type"] not in ["image", "metadata"]:
            if (entry["data_type"] == "output") and (
                entry["tool_id"] == "query_conversation_logs"
            ):
                return_data = "The tool returned conversation logs related to the user's question, which have been removed for brevity"
            else:
                return_data = entry["data"]
            conversation[timestamp] = "{}: {}\n\n".format(sender, return_data)

    conversation_str = ""
    for k, v in sorted(conversation.items()):
        conversation_str += "({}) {}".format(k, v)

    return conversation_str
