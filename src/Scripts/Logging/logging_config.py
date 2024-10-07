import logging


# Define a custom formatter class with ANSI escape codes for colors
class ColorFormatter(logging.Formatter):
    # ANSI escape codes for colors
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35;1m",  # Magenta (bright)
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Get the color for the log level, default to no color
        log_color = self.COLORS.get(record.levelno, self.RESET)
        # Apply the color and reset at the end of the message
        formatted_message = super().format(record)
        return f"{log_color}{formatted_message}{self.RESET}"


# Configure logging
def configure_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create console handler with the custom color formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)
