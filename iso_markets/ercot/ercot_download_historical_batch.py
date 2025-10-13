from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementClickInterceptedException
from datetime import datetime
import time
import os
import glob
import zipfile

def setup_chrome_driver(download_dir):
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": False,
        "safebrowsing.disable_download_protection": True,
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
        "profile.managed_default_content_settings.images": 2
    })
    
    # Add arguments to prevent download interruption
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    
    return webdriver.Chrome(options=chrome_options)

def login(driver, username, password):
    # Wait for the email field and enter the username
    email_field = WebDriverWait(driver, 45).until(
        EC.presence_of_element_located((By.ID, "email"))
    )
    email_field.send_keys(username)

    # Find the password field and enter the password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)

    # Find and click the login button
    login_button = driver.find_element(By.ID, "next")
    login_button.click()

    # Wait for the login process to complete
    WebDriverWait(driver, 10).until(
        EC.url_contains("ercot.com")
    )

    # Wait for and click the "I Agree" button in the dialog
    try:
        agree_button = WebDriverWait(driver, 45).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Agree')]"))
        )
        agree_button.click()
    except TimeoutException:
        print("Timed out waiting for 'I Agree' button. It might not have appeared.")

def wait_for_download_buttons(driver, timeout):
    try:
        download_buttons = WebDriverWait(driver, timeout).until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
        )
        return download_buttons
    except TimeoutException:
        return None

def wait_for_download_modal_to_disappear(driver, timeout=120):
    try:
        # First, check if the modal is present
        modal = driver.find_elements(By.ID, "loading-modal")
        if not modal:
            print("Download modal not found. Continuing...")
            return
        else:
            print("Download modal found. Waiting for it to disappear...")

        # If the modal is present, wait for it to disappear
        WebDriverWait(driver, timeout).until_not(
            EC.presence_of_element_located((By.ID, "loading-modal"))
        )
        print("Download modal disappeared.")
    except TimeoutException:
        print("Download modal did not disappear within the expected time. Continuing...")

def click_button_safely(driver, button, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            # Scroll the button into view
            driver.execute_script("arguments[0].scrollIntoView(true);", button)
            time.sleep(1)  # Allow time for any animations to complete
            
            # Try to click the button
            button.click()
            return True
        except ElementClickInterceptedException:
            if attempt == max_attempts - 1:
                print(f"Failed to click button after {max_attempts} attempts.")
                return False
            print(f"Click intercepted, retrying... (Attempt {attempt + 1})")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred while trying to click the button: {str(e)}")
            return False
    return False

def file_exists_and_not_empty(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


def verify_zip_integrity(download_dir):
    """Check the integrity and contents of downloaded ZIP files"""
    zip_files = glob.glob(os.path.join(download_dir, '*.zip'))
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test the ZIP file integrity
                zip_ref.testzip()
                
                # List contents
                file_list = zip_ref.namelist()
                total_size = sum(zip_ref.getinfo(name).file_size for name in file_list)
                
                print(f"ZIP: {os.path.basename(zip_path)}")
                print(f"  Files: {len(file_list)}")
                print(f"  Total uncompressed size: {total_size:,} bytes")
                print(f"  ZIP file size: {os.path.getsize(zip_path):,} bytes")
                
                # Check if ZIP seems incomplete (too small or only 1 file when expecting more)
                if len(file_list) == 1 and total_size < 1000:  # Less than 1KB uncompressed
                    print(f"  WARNING: {os.path.basename(zip_path)} may be incomplete!")
                    
        except zipfile.BadZipFile:
            print(f"ERROR: {os.path.basename(zip_path)} is corrupted!")
        except Exception as e:
            print(f"ERROR checking {os.path.basename(zip_path)}: {e}")

def check_downloads_complete(download_dir, timeout=600):
    # Wait for a short time to allow downloads to start
    time.sleep(1)
    
    # Check for any .crdownload or .tmp files
    incomplete_downloads = glob.glob(os.path.join(download_dir, '*.crdownload')) + glob.glob(os.path.join(download_dir, '*.tmp'))
    
    # Wait up to timeout seconds for downloads to complete (default 10 minutes)
    start_time = time.time()
    while incomplete_downloads and (time.time() - start_time) < timeout:
        print(f"Waiting for downloads to complete... {len(incomplete_downloads)} files still downloading")
        time.sleep(1)
        incomplete_downloads = glob.glob(os.path.join(download_dir, '*.crdownload')) + glob.glob(os.path.join(download_dir, '*.tmp'))
    
    return len(incomplete_downloads) == 0

def scrape_and_download_batch(url, download_dir, skipPages=None):
    print(f"Fetching historical data: {download_dir} \n{url}")
    os.makedirs(download_dir, exist_ok=True)
    driver = setup_chrome_driver(download_dir)
    driver.get(url)
    login(driver, "enrico.ladendorf@battalion.energy", "52eDnRryUrA5fN6")

    # Wait for the page to load
    WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
    )

    # Select 500 items per page:
    #<select class="form-select me-2 w-auto"><option value="10">10 per page</option><option value="25">25 per page</option><option value="50">50 per page</option><option value="100">100 per page</option><option value="200">200 per page</option><option value="500">500 per page</option></select>
    try:
        select_element = WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select.me-2.w-auto"))
        )
        select = Select(select_element)
        select.select_by_value("500")
        
        time.sleep(1)  # Wait for buttons to disappear and reappear
    except Exception as e:
        print(f"Failed to select 500 items per page: {str(e)}")
        
        #Find All Checkbox and check it: <input type="checkbox" class="form-check-input" id="checkAllBox" value="All">
    checkbox = WebDriverWait(driver, 45).until(
        EC.element_to_be_clickable((By.ID, "checkAllBox"))
    )
    checkbox.click()
    time.sleep(1)

    page_count = 0
    while True:
        #Wait for the download buttons to appear
        download_buttons = wait_for_download_buttons(driver, 120)
        if not download_buttons:
            print("Failed to find download buttons. Exiting.")
            break
        
        # Check if we've reached the specified number of pages to skip
        while skipPages is not None and page_count <= skipPages:
            print(f"Page Count: {page_count}")
            # Click the next page button >>
            try:
                next_button = WebDriverWait(driver, 45).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[@class='page-link' and text()='»']"))
                )
                if not click_button_safely(driver, next_button):
                    print("Failed to click next page button. Exiting.")
                    break
                time.sleep(2)  # Wait for buttons to disappear and reappear
                page_count += 1
            except (NoSuchElementException, TimeoutException):
                print("No more pages to scrape or timed out waiting for next page.")
                break

        # Wait for the checkbox to be clickable
        checkbox = WebDriverWait(driver, 45).until(
            EC.element_to_be_clickable((By.ID, "checkAllBox"))
        )

        # Check if the checkbox is already selected
        if not checkbox.is_selected():
            # If it's not selected, click it
            checkbox.click()
            time.sleep(1)
        
        try:
            # Wait for the Download button to be clickable
            download_button = WebDriverWait(driver, 45).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
            )
            
            # Click the button
            download_button.click()
            
            print("Download button clicked successfully.")
            
            # Wait longer for server to prepare the download
            time.sleep(30)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
        wait_for_download_modal_to_disappear(driver, timeout=300)

        # Check if downloads are complete
        if check_downloads_complete(download_dir):
            print("Downloads completed successfully.")
        else:
            print("Downloads may not have completed. Please check the download directory.")


        # Click the next page button >>
        try:
            next_button = WebDriverWait(driver, 45).until(
                EC.element_to_be_clickable((By.XPATH, "//span[@class='page-link' and text()='»']"))
            )
            if not click_button_safely(driver, next_button):
                print("Failed to click next page button. Exiting.")
                break
            time.sleep(2)  # Wait for buttons to disappear and reappear
        except (NoSuchElementException, TimeoutException):
            print("No more pages to scrape or timed out waiting for next page.")
            break

    driver.quit()


def scrape_and_download_by_date(url, download_dir, end_date="2024-08-24", start_date="2010-01-01" ,pageSize=500):
    """
    Scrape and download data using date filtering instead of pagination.
    
    Args:
        url: The ERCOT data URL
        download_dir: Directory to save downloads
        end_date: End date in YYYY-MM-DD format (default: "2024-08-24")
        start_date: Start date in YYYY-MM-DD format (default: "2010-01-01")
    """
    # Convert dates from YYYY-MM-DD to MM/DD/YYYY format for the website
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    start_date_formatted = start_date_obj.strftime("%m/%d/%Y")
    end_date_formatted = end_date_obj.strftime("%m/%d/%Y")
    
    print(f"Fetching historical data by date range: {start_date_formatted} to {end_date_formatted}")
    print(f"Download directory: {download_dir}")
    print(f"URL: {url}")
    
    os.makedirs(download_dir, exist_ok=True)
    driver = setup_chrome_driver(download_dir)
    driver.get(url)
    login(driver, "enrico.ladendorf@battalion.energy", "52eDnRryUrA5fN6")

    # Wait for the page to load
    WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
    )

    # Select 500 items per page
    try:
        select_element = WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select.form-select.me-2.w-auto"))
        )
        select = Select(select_element)
        select.select_by_value(str(pageSize))
        time.sleep(2)  # Wait for buttons to disappear and reappear
    except Exception as e:
        print(f"Failed to select 500 items per page: {str(e)}")

    # Set the start date (first date input)
    try:
        start_date_input = WebDriverWait(driver, 45).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='date']"))
        )
        
        # Use simple send_keys which has been proven to work
        start_date_input.clear()
        start_date_input.send_keys(start_date_formatted)
        start_date_input.send_keys(Keys.TAB)
        
        print(f"Set start date to: {start_date_formatted}")
        time.sleep(1)
    except Exception as e:
        print(f"Failed to set start date: {str(e)}")

    # Set the end date (second date input)
    try:
        end_date_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='date']")
        if len(end_date_inputs) >= 2:
            end_date_input = end_date_inputs[1]  # Second date input
            
            # Use simple send_keys which has been proven to work
            end_date_input.clear()
            end_date_input.send_keys(end_date_formatted)
            end_date_input.send_keys(Keys.TAB)
            
            print(f"Set end date to: {end_date_formatted}")
            time.sleep(1)
        else:
            print("Could not find second date input for end date")
    except Exception as e:
        print(f"Failed to set end date: {str(e)}")

    # Click the Update button
    try:
        # Wait a moment for the button to become enabled after date input
        time.sleep(1)
        
        update_button = WebDriverWait(driver, 45).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@class='btn btn-info' and text()='Update']"))
        )
        
        # Double-check the button is enabled before clicking
        if update_button.is_enabled():
            update_button.click()
            print("Clicked Update button successfully")
        else:
            print("Update button found but not enabled - trying alternative click method")
            driver.execute_script("arguments[0].click();", update_button)
            print("Clicked Update button via JavaScript")
        
        time.sleep(3)  # Wait for the form to update
    except Exception as e:
        print(f"Failed to click Update button: {str(e)}")

    # Wait for the page to reload and download buttons to appear
    WebDriverWait(driver, 120).until(
        EC.presence_of_element_located((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
    )

    page_count = 0
    while True:
        # Wait for the All checkbox to be clickable
        try:
            checkbox = WebDriverWait(driver, 45).until(
                EC.element_to_be_clickable((By.ID, "checkAllBox"))
            )
            
            # Check if the checkbox is already selected
            if not checkbox.is_selected():
                checkbox.click()
                time.sleep(1)
                print("Selected All checkbox")
        except Exception as e:
            print(f"Failed to select All checkbox: {str(e)}")
            break

        #time.sleep(1)
        # Click the Download button
        try:
            download_button = WebDriverWait(driver, 45).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@class='btn btn-link align-baseline' and contains(., 'Download')]"))
            )
            #download_button.click()
            click_button_safely(driver, download_button)
            print(f"Page {page_count + 1}: Download button clicked successfully.")
            
            # Wait longer for server to prepare the download  
            #time.sleep(5)
        except Exception as e:
            print(f"Failed to click download button: {str(e)}")
            break

        # Wait for the download modal to disappear
        #time.sleep(2)
        wait_for_download_modal_to_disappear(driver, timeout=300)

        # Check if downloads are complete
        if check_downloads_complete(download_dir):
            print(f"Page {page_count + 1}: Downloads completed successfully.")
        else:
            print(f"Page {page_count + 1}: Downloads may not have completed. Please check the download directory.")
        #time.sleep(3)

        # Try to go to the next page with exponential backoff
        wait_time = 5  # Start with 5 seconds
        max_wait_time = 120  # Max 2 minutes
        max_attempts = 1000  # Maximum attempts before giving up on this page
        
        for attempt in range(max_attempts):
            try:
                next_button = WebDriverWait(driver, 45).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[@class='page-link' and text()='»']"))
                )
                if click_button_safely(driver, next_button):
                    time.sleep(1)  # Wait for buttons to disappear and reappear
                    page_count += 1
                    print(f"Successfully moved to page {page_count + 1}")
                    break  # Success, move to next iteration
                else:
                    print(f"Failed to click next page button (attempt {attempt + 1}/{max_attempts}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time = min(wait_time * 2, max_wait_time)  # Exponential backoff, max 2 minutes
            except (NoSuchElementException, TimeoutException):
                print(f"No more pages to scrape or timed out waiting for next page (attempt {attempt + 1}/{max_attempts}). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, max_wait_time)  # Exponential backoff, max 2 minutes
        else:
            # If we've exhausted all attempts, wait the max time and try again
            print(f"Exhausted all attempts for page {page_count + 1}. Waiting {max_wait_time} seconds before retrying...")
            time.sleep(max_wait_time)
            # Don't break - keep trying infinitely

    # Keep the browser open indefinitely for human intervention
    print("Script completed but keeping browser open for manual intervention...")
    print("Press Ctrl+C to close the browser manually.")
    try:
        while True:
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("Manual interruption detected. Closing browser...")
        driver.quit()


def findNextDownload(download_directory):
    try:
        # Find all files in the download directory with the specified pattern
        files = glob.glob(os.path.join(download_directory, '*.DAMHRLMPNP4183_csv.zip'))

        # Sort the files by name in descending order
        files.sort(reverse=True)

        if files:
            # Get the most recent file name
            most_recent_file = files[0]

            # Extract the date string from the file name
            date_string = os.path.basename(most_recent_file).split('.')[3]

            # Return the date string in the format YYYYMMDD
            return date_string
    except Exception as e:
        print(f"Could not find last file, starting from the end.")
        return None

    return None

if __name__ == "__main__":
        # Set download directory based on operating system / user home directory
    user_home_dir = os.path.expanduser("~")  # Returns '/Users/<name>' on macOS and '/home/<name>' on Linux
    download_directory = os.path.join(user_home_dir, "data", "ERCOT_data") + os.sep


    scrape_and_download_by_date("https://data.ercot.com/data-product-archive/np4-732-cd", 
            download_directory + "Wind Power Production - Hourly Averaged Actual and Forecasted Values",
            end_date="2025-10-10", 
            start_date="2010-01-01")

    scrape_and_download_by_date("https://data.ercot.com/data-product-archive/np4-737-cd", 
            download_directory + "Solar Power Production - Hourly Averaged Actual and Forecasted Values",
            end_date="2025-10-10", 
            start_date="2010-01-01")


    scrape_and_download_by_date("https://data.ercot.com/data-product-archive/np3-560-cd", 
            download_directory + "Seven-Day Load Forecast by Forecast Zone",
            end_date="2025-10-10", 
            start_date="2010-01-01")

    scrape_and_download_by_date("https://data.ercot.com/data-product-archive/np4-193-cd", 
            download_directory + "DAM Total Energy Sold",
            end_date="2025-10-10", 
            start_date="2010-01-01")


    #TODO: Grab all the historical data listed here: 

    #RT Market SPPs and  LMPs:
    # Use the new date-based method instead of skip pages
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP6-788-CD", 
     #                          download_directory + "LMPs_by_Resource_Nodes,_Load_Zones_and_Trading_Hubs",
     #                          end_date="2024-08-24", 
     #                          start_date="2015-01-01")
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP6-905-CD", 
     #                          download_directory + "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
     #                          end_date="2024-08-25", 
     #                         start_date="2011-01-01",pageSize=500)
    
    #DAM Hourly SPPs and LMPs:
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP4-183-CD",
    #                        download_directory + "DAM_Hourly_LMPs",
    #                        end_date="2025-09-01", 
    #                        start_date="2009-01-01",pageSize=100)
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP4-190-CD", 
    #                            download_directory + "DAM_Settlement_Point_Prices",
    #                        end_date="2024-08-29", 
    #                        start_date="2009-01-01",pageSize=500)
    
    #Ancillary Services:
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP4-188-CD", 
    #                           download_directory + "DAM_Clearing_Prices_for_Capacity",
    #                           end_date="2024-08-24", 
    #                           start_date="2009-01-01")
    
    #Shadow Prices /congestion:
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP6-86-CD", 
    #                           download_directory + "SCED_Shadow_Prices_and_Binding_Transmission_Constraints",
    #                           end_date="2025-07-28", 
    #                           start_date="2009-01-01")
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP4-191-CD", 
    #                           download_directory + "DAM_Shadow_Prices",
    #                           end_date="2025-07-28", 
    #                           start_date="2009-01-01")
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP6-787-CD",
    #                       download_directory + "LMPs_by_Electrical_Bus",
    #                        end_date="2022-08-01", 
    #                        start_date="2009-01-01",pageSize=500)
       
    
    #60 Day Disclosure Reports:
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP3-965-ER", 
    #                           download_directory + "60-Day_SCED_Disclosure_Reports",
    #                           end_date="2023-08-29", 
    #                          start_date="2011-12-30",pageSize=25)

    
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP1-301", 
    #                           download_directory + "60-Day_COP_Adjustment_Period_Snapshot",
    #                           end_date="2023-08-29", 
    #                          start_date="2011-12-30",pageSize=25)
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP3-991-EX", 
    #                           download_directory + "60-Day_COP_All_Updates",
    #                           end_date="2023-08-29", 
    #                          start_date="2011-12-30",pageSize=25)
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP3-990-EX", 
    #                           download_directory + "60-Day_SASM_Disclosure_Reports",
    #                           end_date="2023-08-29", 
    #                          start_date="2011-12-30",pageSize=25)
    
    #scrape_and_download_by_date("https://data.ercot.com/data-product-archive/NP3-966-ER", 
    #                           download_directory + "60-Day_DAM_Disclosure_Reports",
    #                           end_date="2016-11-11", 
    #                          start_date="2011-12-30",pageSize=25)
    
    
    
    
