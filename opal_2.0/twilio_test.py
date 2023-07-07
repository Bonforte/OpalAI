from twilio.rest import Client

account_sid = 'ACf0eb4d916265affb03fbbc391a151fbe'
auth_token = '3138f8237f5adf25b073925a65a7a139'
client = Client(account_sid, auth_token)

from_whatsapp_number = 'whatsapp:+14155238886'  # Your Twilio phone number
to_whatsapp_number = 'whatsapp:+40734260029'  # Recipient's phone number


client.messages.create(
    body='Success',
    from_=from_whatsapp_number,
    to=to_whatsapp_number
)