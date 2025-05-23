import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import shutil
from selenium.webdriver.chrome.service import Service

SHIHU_URL = "http://www.tcmip.cn/ETCM/index.php/Home/Index/yc_details.html?id=295"
OUTPUT_FILE = "shihu_components.xlsx"

PAGE_LOAD_TIMEOUT = 30  
ELEMENT_WAIT_TIMEOUT = 15 

def main():
    chromedriver_path = shutil.which("chromedriver")
    if not chromedriver_path:
        raise RuntimeError(
            "chromedriver executable not found in PATH. "
            "Please install chromedriver (e.g., brew install chromedriver)."
        )
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'eager'
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    try:
        try:
            driver.get(SHIHU_URL)
        except TimeoutException:
            raise RuntimeError(f"Timed out loading {SHIHU_URL!r} after {PAGE_LOAD_TIMEOUT}s")
        wait = WebDriverWait(driver, ELEMENT_WAIT_TIMEOUT)

        # Load component links from static table or fallback to accordion panel
        try:
            table = wait.until(EC.presence_of_element_located((By.ID, "comta")))
            component_links = table.find_elements(By.CSS_SELECTOR, "td a.tdcolor")
        except TimeoutException:
            comp_header = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//a[text()='Components']"))
            )
            comp_header.click()
            component_links = wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#L7713 + div a.tdcolor"))
            )

        results = []

        for link in component_links:
            name = link.text.strip()
            href = link.get_attribute("href")
            if not href:
                continue

            # Navigate to the component page
            driver.get(href)

            # 5) Extract the four fields
            def get_text_by_label(label_text):
                try:
                    table = driver.find_element(By.ID, "table")
                    row = table.find_element(
                        By.XPATH, f".//tr[td[normalize-space()='{label_text}']]"
                    )
                    return row.find_element(By.XPATH, "td[2]").text.strip()
                except NoSuchElementException:
                    driver.execute_script(
                        "arguments[0].scrollIntoView(true);",
                        driver.find_element(By.XPATH, f"//div[text()='{label_text}']")
                    )
                    val_div = driver.find_element(
                        By.XPATH, f"//div[text()='{label_text}']/following-sibling::div"
                    )
                    return val_div.text.strip()

            eng_name = get_text_by_label("Ingredient Name in English")
            formula  = get_text_by_label("Molecular Formula")
            weight   = get_text_by_label("Molecular Weight")

            try:
                table = driver.find_element(By.ID, "table")
                row = table.find_element(
                    By.XPATH, ".//tr[td[normalize-space()='2D-Structure']]"
                )
                img = row.find_element(By.XPATH, "td[2]//img")
            except NoSuchElementException:
                img = driver.find_element(
                    By.XPATH,
                    "//div[text()='2D-Structure']/following-sibling::div//img"
                )
            structure_url = img.get_attribute("src")

            results.append({
                "Component Link Text": name,
                "Ingredient Name (EN)": eng_name,
                "2D-Structure URL": structure_url,
                "Molecular Formula": formula,
                "Molecular Weight": weight,
            })

            # go back to the herb page (or reopen it)
            driver.back()
            
            # re-expand the Components panel
            wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//a[text()='Components']"))).click()
            time.sleep(0.2)

        # 6) Dump to Excel
        df = pd.DataFrame(results)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Done! {len(df)} components written to {OUTPUT_FILE!r}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()