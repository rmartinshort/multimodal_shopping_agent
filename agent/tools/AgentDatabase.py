import tinydb
import uuid
import datetime
import logging

logger = logging.getLogger(__name__)


class AgentDatabase:
    """
    A class to manage storing and retrieving images and text with unique IDs in TinyDB.
    """

    def __init__(self, db_file="agent_db.json"):
        """
        Initializes the database connection.

        Args:
            db_file (str, optional): Path to the TinyDB database file. Defaults to "agent_db.json".
        """
        self.db_file = db_file
        self.db = tinydb.TinyDB(self.db_file)

    def _generate_unique_id(self):
        """Generates a unique ID using UUID.

        Returns:
            str: A unique UUID string.
        """
        return str(uuid.uuid4())

    def store_image(self, user_id, conversation_id, tool_id, image_data):
        """
        Stores image data in the database.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.
            tool_id (str): The ID of the tool used to generate the image.
            image_data (bytes): The image data to store.

        Returns:
            str: The unique ID of the stored image data, or None if storage failed.
        """
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
        """
        Stores text data in the database.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.
            tool_id (str): The ID of the tool used to generate the text.
            data_type (str): The type of text data (e.g., "query", "response").
            text_data (str): The text data to store.

        Returns:
            str: The unique ID of the stored text data, or None if storage failed.
        """
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
            dict: The data dictionary if found, None otherwise.  Returns None if not found.
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
        Retrieves data by user ID.

        Args:
            user_id (str): The ID of the user.
            remove_image_data (bool, optional): Whether to exclude image data from the results. Defaults to True.

        Returns:
            list: A list of data dictionaries associated with the user ID.  Returns an empty list if no data is found.
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

        Args:
            unique_id (str): The unique ID of the data to delete.
        """
        data = self.get_data_by_message_id(unique_id)
        if data:
            Data = tinydb.Query()
            self.db.remove(Data.id == unique_id)
            logger.info(f"Data with ID {unique_id} deleted.")
        else:
            logger.info(f"Data with ID {unique_id} not found, deletion skipped.")
