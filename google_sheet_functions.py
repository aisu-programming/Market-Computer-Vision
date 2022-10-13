from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = [ "https://www.googleapis.com/auth/spreadsheets" ]


def get_credentials(function):
    def wrapper(*args):
        credentials = None
        if os.path.exists("token.json"):
            credentials = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                credentials = flow.run_local_server(port=0)
            with open("token.json", 'w') as token:
                token.write(credentials.to_json())
        function(*args, credentials)
    return wrapper


@get_credentials
def create_sheet(spreadsheet_id, title, credentials):
    try:
        service = build("sheets", "v4", credentials=credentials)
        request_body = { "requests": [{ "addSheet": { "properties": { "title": title }}}]}
        result = service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=request_body).execute()
        return result
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise error


@get_credentials
def read_values(spreadsheet_id, range_name, credentials):
    try:
        service = build("sheets", "v4", credentials=credentials)
        result = service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
        rows = result.get("values", [])
        # for row in rows: print(row)
        return result
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error


@get_credentials
def append_values(spreadsheet_id, range_name, values, credentials):
    try:
        service = build("sheets", "v4", credentials=credentials)
        body = { "values": values }
        result = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id, range=range_name,
            valueInputOption="USER_ENTERED", body=body).execute()
        # print(f"{(result.get('updates').get('updatedCells'))} cells appended.")
        return result
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error


if __name__ == "__main__":

    TEST_SPREADSHEET_ID = "1cJNbeULQvetY2LEde1RDGsu_31JY_Av_AQMNBkaAvWQ"
    TEST_RANGE_NAME = "2022/10/11"

    create_sheet(TEST_SPREADSHEET_ID, "test")
    # read_values(TEST_SPREADSHEET_ID, TEST_RANGE_NAME)
    # append_values(TEST_SPREADSHEET_ID, TEST_RANGE_NAME, [[ 1, 2, 3, 4, 5 ]])