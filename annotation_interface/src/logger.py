import os
import logging

class Logger:
    def __init__(self):

        logs_dir = './logs'
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = f'{logs_dir}/app.log'
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def info(self, message):
        logging.info(message)

    def error(self, message):
        logging.error(message)