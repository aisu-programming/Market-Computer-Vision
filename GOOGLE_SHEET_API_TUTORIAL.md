# Google Sheet API tutorial
1. Switch to the company's Google account.

2. Enable the API and Authorize credentials for a desktop application as the tutorial in here:
   https://developers.google.com/sheets/api/quickstart/python
   > <img src="https://i.imgur.com/43YFmQf.png" width="600" />

3. Save the JSON file to the directory *Market-Computer-Vision* and rename it to `credentials.json`.

4. Create a Google Sheet and by your account so that you have the authority to edit it.
   Copy its ID from URL.
   > <img src="https://i.imgur.com/Zon728X.png" width="800" />
   > For this example, the ID is *1cJNbeULQvetY2LEde1RDGsu31JYAvAQMNBkaAvWQ*.
   
   And add this ID into file _.env_.
   > <img src="https://user-images.githubusercontent.com/66176726/196001362-24e4c828-cb4a-4fbd-9b72-be200ecd67d1.png" width="600" />

5. At the first time of the execution with the `--save-google-sheet` option, the browser would pop out a login request page:
   > <img src="https://i.imgur.com/TioEKmd.png" width="400" />

   Login with the company account.
   
6. After login, you should see this page.
   > <img src="https://i.imgur.com/nt5ALPA.png" width="500" />

   Click the "Advanced" button at the bottom-left.
   
7. Then click the "Go to {Project Name} (unsafe)" button.
   > <img src="https://i.imgur.com/fMFiV0f.png" width="500" />

8. Then you should see this page. Click the "Continue" button.
   > <img src="https://i.imgur.com/Kt0FoEk.png" width="400" />

9. If you see this page, then it's done.
   > <img src="https://i.imgur.com/7a9xYFw.png" width="500" />
