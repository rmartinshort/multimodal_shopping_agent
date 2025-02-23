import tinydb
import datetime
import uuid
import logging

logger = logging.getLogger("agent_tools")
logger.setLevel(logging.INFO)


class AgentDatabase:
    """
    A class to manage storing and retrieving images and text with unique IDs in TinyDB.
    """

    def __init__(self, db_file="agent_db.json"):
        """
        Initializes the database connection and image directory.

        Args:
            db_file (str): Path to the TinyDB database file.
            image_dir (str): Directory to save image files.
        """
        self.db_file = db_file
        self.db = tinydb.TinyDB(self.db_file)

    def _generate_unique_id(self):
        """Generates a unique ID using UUID."""
        return str(uuid.uuid4())

    def store_image(self, user_id, conversation_id, tool_id, image_data):
        unique_id = self._generate_unique_id()
        timestamp = str(datetime.datetime.now())

        data = {
            "id": unique_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "tool_id": tool_id,
            "data_type": "image",
            "data": image_data,
        }
        try:
            self.db.insert(data)
            logger.info(f"Data stored with ID: {unique_id}")
            return unique_id
        except Exception as e:
            logger.info(f"An error occurred during the store: {e}")
            return None

    def store_text(self, user_id, conversation_id, tool_id, data_type, text_data):
        unique_id = self._generate_unique_id()
        timestamp = str(datetime.datetime.now())

        data = {
            "id": unique_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "tool_id": tool_id,
            "data_type": data_type,
            "data": text_data,
        }
        try:
            self.db.insert(data)
            logger.info(f"Data stored with ID: {unique_id}")
            return unique_id
        except Exception as e:
            logger.info(f"An error occurred during the store: {e}")
            return None

    def get_data_by_message_id(self, unique_id):
        """
        Retrieves data by unique ID.

        Args:
            unique_id (str): The unique ID of the data to retrieve.

        Returns:
            dict: The data dictionary if found, None otherwise.
        """
        Data = tinydb.Query()
        result = self.db.search(Data.id == unique_id)
        if result:
            return result[0]  # Return the first (and only) matching document
        else:
            logger.info(f"No data found with ID: {unique_id}")
            return None

    def get_data_by_user_id(self, user_id, remove_image_data=True):
        """
        Retrieves data by unique ID.

        Args:
            unique_id (str): The unique ID of the data to retrieve.

        Returns:
            dict: The data dictionary if found, None otherwise.
        """
        if remove_image_data:
            result = self.db.search(
                (tinydb.Query().user_id == user_id)
                & (tinydb.Query().data_type != "image")
            )
        else:
            result = self.db.search(tinydb.Query().user_id == user_id)
        if result:
            return result
        else:
            logger.info(f"No data found with user ID: {user_id}")
            return []

    def delete_data(self, unique_id):
        """
        Deletes data by unique ID.
        """
        data = self.get_data_by_message_id(unique_id)
        if data:
            Data = tinydb.Query()
            self.db.remove(Data.id == unique_id)
            logger.info(f"Data with ID {unique_id} deleted.")
        else:
            logger.info(f"Data with ID {unique_id} not found, deletion skipped.")
