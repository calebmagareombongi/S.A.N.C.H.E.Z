import imaplib
import smtplib
import email
from email.header import decode_header
import ssl
import speech_recognition as sr
from speech_code import speak_save, recognize_speech_save, speak_no_save, cleanup_old_files  # Import functions from speech_code.py

# Define your email account details
IMAP_SERVER = 'imap.gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
EMAIL_ACCOUNT = 'calebmagareombongi@gmail.com'
PASSWORD = 'bseq qdpc uhna mpyt'

# Connect to IMAP server to read emails
def read_emails():
    try:
        # Connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, PASSWORD)
        
        # Select the mailbox you want to check (inbox by default)
        mail.select('inbox')

        # Search for all unread emails
        status, messages = mail.search(None, '(UNSEEN)')
        
        # Convert messages to a list of email IDs
        email_ids = messages[0].split()
        
        # Process each email
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            
            # Extract the message content
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg['Subject'])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding if encoding else 'utf-8')
                    
                    # From where the email came
                    sender = msg.get('From')

                    # Speak out the email subject and sender
                    speak_save(f'Reading email from {sender} with subject: {subject}')
                    
                    # If the email message is multipart
                    if msg.is_multipart():
                        for part in msg.walk():
                            # Extract the plain text part
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                speak_save(f"Message Body: {body}")
                    else:
                        # Extract content if it's not multipart
                        body = msg.get_payload(decode=True).decode()
                        speak_save(f"Message Body: {body}")
        
        # Close the connection and logout
        mail.close()
        mail.logout()
    
    except Exception as e:
        speak_save(f'Error reading emails: {e}')

# Function to send an email
def send_email(subject, body, to_email):
    try:
        # Set up the server
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, 465, context=context) as server:
            server.login(EMAIL_ACCOUNT, PASSWORD)
            
            # Create the email
            msg = f'Subject: {subject}\n\n{body}'
            
            # Send the email
            server.sendmail(EMAIL_ACCOUNT, to_email, msg)
            speak_save(f'Email sent to {to_email}')
    
    except Exception as e:
        speak_save(f'Error sending email: {e}')

# Speech recognition for email management
def email_management():
    # Ask for sending or reading emails
    speak_save("Do you want to send or read emails? Please wait a moment and then speak.")
    
    # Listen for the user's command
    email_command = recognize_speech_save()

    # Check if the user wants to send an email
    if email_command and "send" in email_command.lower():
        # Ask for the subject of the email
        speak_save("Please provide the subject.")
        subject = recognize_speech_save()
        speak_save(f"Subject: {subject}")

        # Ask for the body of the email
        speak_save("Now, please provide the body of the email.")
        body = recognize_speech_save()
        speak_save(f"Body: {body}")

        # Ask for the recipient's email address
        speak_save("Please provide the recipient's email address.")
        to_email = recognize_speech_save()
        speak_save(f"Recipient: {to_email}")
        
        # Send the email
        send_email(subject, body, to_email)
    
    # Otherwise, read emails
    elif email_command and "read" in email_command.lower():
        read_emails()
    else:
        speak_save("Sorry, I didn't catch that. Please say 'send' or 'read'.")

# Example usage
if __name__ == '__main__':
    email_management()
