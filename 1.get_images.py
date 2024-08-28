import time

from playwright.sync_api import sync_playwright
num = 0
with sync_playwright() as p:
    browser = p.firefox.launch(headless=True)
    page = browser.new_page()
    page.goto('')
    page.locator('xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[1]/div/ul/li[11]').click()
    time.sleep(2)
    page.locator('xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[2]/div[2]/ul/li[11]/div[2]/div/div[1]/div[3]').click()
    time.sleep(2)
    while True:
        try:
            for i in range(5):
                print("抓取抓取的图片数量：{}".format(str(num)))
                page.locator('xpath=//*[@id="dx_captcha_clickword_hits_4"]').screenshot(path="./images/{}.png".format(str(num)))
                #/html/body/div[11]/div/div[6]/div[2]/span[2]/img
                page.locator('xpath=/html/body/div[11]/div/div[6]/div[2]/span[2]/img').click()
                num += 1
                time.sleep(2)
            time.sleep(2)
            page.reload()
            time.sleep(4)
            page.locator('xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[1]/div/ul/li[11]').click()
            time.sleep(2)
            page.locator(
                'xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[2]/div[2]/ul/li[11]/div[2]/div/div[1]/div[3]').click()
        except Exception as e:
            print("err:", e)
            print("出错了，重新启动浏览器！")
            time.sleep(5)
            browser.close()
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto('')
            page.locator('xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[1]/div/ul/li[11]').click()
            time.sleep(2)
            page.locator(
                'xpath=/html/body/div[1]/div[2]/div/div[6]/div/div[2]/div[2]/div[2]/ul/li[11]/div[2]/div/div[1]/div[3]').click()
            time.sleep(2)