import os
from dotenv import load_dotenv
load_dotenv()

from twilio.rest import Client

import streamlit as st

account_sid = st.secrets['TWILIO_ACCOUNT_SID']
auth_token = st.secrets['TWILIO_AUTH_TOKEN']
messaging_service_sid = st.secrets['MESSAGE_SERVICE_SID']
def send_message(message_body, number):
    client = Client(account_sid, auth_token)
    
    message_body = client.messages.create(  
                                messaging_service_sid=messaging_service_sid, 
                                body=message_body,      
                                to=number 
                            ) 
    
    print('Message Sent!')


if __name__ == '__main__':
    send_message()