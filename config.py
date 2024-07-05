import configparser

config = configparser.ConfigParser()
config.read("config.ini")
api_key = config["AZURE_OPENAI"]["AZURE_OPENAI_API_KEY"]
endpoint = config["AZURE_OPENAI"]["AZURE_OPENAI_ENDPOINT"]
tavily_api_key = config["TAVILY"]["TAVILY_API_KEY"]
