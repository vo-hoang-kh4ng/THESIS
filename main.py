import os
from dotenv import load_dotenv
from my_utils import run_social_media_monitoring

# Load các biến môi trường từ .env
load_dotenv()

    
if __name__ == "__main__":
    print("Welcome to the Social Media Monitoring Crew!")
    brand_name = input("Enter the name of the brand or influencer you want to research: ")
    
    result = run_social_media_monitoring(brand_name)
    
    if result:
        print("\n" + "="*50 + "\n")
        print("Final Report:")
        print(result)
    else:
        print("Failed to generate the report. Please try again later.")