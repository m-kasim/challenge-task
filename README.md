# Challenge: TensorFlow Inference API via Django REST Framework
This project demonstrates a Django REST Framework app utilizing a trained `TensorFlow` model to classify scientific articles category based on their `abstract`. It utilizes fine-tuned SciBERT model to perform `multi-label classification`. The project is deployed on a real VPS on Amazon EC2 and it features robust input validation, efficient text preprocessing, and detailed error handling, with production-grade code structure including modular components like preprocessors and serializers. The project includes unit tests for serializer validation and API endpoint functionality, ensuring reliability and maintainability.

### Quick start
Follow these, if you just want to run the demo of the project:

1. Connect to machine with provided secret access key:
```
ssh admin@52.29.200.255
```
2. Activate Django virtual environment:
```
source /home/admin/databases/task_api_virtual_environment/bin/activate && cd /home/admin/databases/task_api_virtual_environment/
```
3. Run demo web server:
```
python3 manage.py runserver 0.0.0.0:8000
```

4. Send inference request to DRF API:
Send a POST request continingabstract to the API endpoint via `cURL` (https://curl.se/docs/install.html).
Currently it takes roughly around 5-7s to produce a response, which is quite good for a machine relying only on a standard 2-core CPU:
- API endpoint: `http://52.29.200.255:8000/predict/`
- cURL query:
```
curl -X POST -H "Content-Type: application/json" -d '{"abstract": "A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy. The region of phase space is specified in which the calculation is most reliable. Good agreement is demonstrated with data from the Fermilab Tevatron, and predictions are made for more detailed tests with CDF and DO data. Predictions are shown for distributions of diphoton pairs produced at the energy of the Large Hadron Collider (LHC). Distributions of the diphoton pairs from the decay of a Higgs boson are contrasted with those produced from QCD processes at the LHC, showing that enhanced sensitivity to the signal can be obtained with judicious selection of events."}' http://52.29.200.255:8000/api/predict/
```

5. Response (probability distribution of classes):

NOTE: This is an example for multi-label classification response (with no threshold applied, otherwise a threshold of `0.50` is perhaps a good one)
```
{ "predictions":
[
 [0.8841670155525208,0.06523384153842926,0.28387996554374695,0.4605664312839508,0.8175318837165833,
  0.6500208973884583,0.6196727752685547,0.38454583287239075,0.9463230967521667,0.28972217440605164,
  0.5458275675773621,0.506781816482544,0.7191378474235535,0.7842949032783508,0.8380508422851562,
  0.6719673871994019,0.9093359708786011,0.19130639731884003]
]
}
```
---

### Setup & Installation details
The project is tested with and requires TensorFlow `2.17.0`:
```
sudo apt-get install python3.11-venv
pip3 install tensorflow==2.17.0
pip3 install transformers
pip3 install tf_keras
```

---

### How to the setup Django project:
`
python3 -m venv task_api_virtual_environment
`

`
source task_api_virtual_environment/bin/activate
`

`
python -m pip install Django
`

`
python -m pip install djangorestframework
`

`
django-admin startproject django_project .
`

`
cd django_project
`

`
django-admin startapp api_demo
`

`
cd ..
`

---

### Run migrations
`
python manage.py migrate
`

You should see the output:
```
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying auth.0012_alter_user_first_name_max_length... OK
  Applying sessions.0001_initial... OK
```
### Create Django admin user (will ask for password):
`
python manage.py createsuperuser --username admin --email admin@example.com
`

---

### Run server:
Make sure you are in the same directory as `manage.py`:

**NOTE:** Please mind the port `8000` - if you miss it, or try to use a different one, you will be blocked by the AWS firewall!
`
python3 manage.py runserver 0.0.0.0:8000
`

---

### Live deploymnet
To deploy the server, we're setting up a new AWS EC2 instance.
We are running the same commands as in the beginning of this document.
To connect to the instance type from your terminal (assuming you are on a UNIX based machine or if you are on Windows, you have to download CygWin):

**NOTE:** This instance doesn't have a GPU available, because leaving a GPU-powered instance running is too expensive even for demo purposes.

Command:
```ssh admin@52.29.200.255```

We update the instance and install some required packages:
```
sudo apt-get update && sudo apt-get install vim wget htop sudo unzip curl git
```

We also add some 2GB swap space to the instance, because otherwise TensorFlow will crash even on install (follow the guide):
```
https://web.archive.org/web/20171228135623/https://www.godaddy.com/help/add-memory-swap-ubuntu-or-debian-7-17326
```

Activate the virtual environment:
```
source /home/admin/databases/task_api_virtual_environment/bin/activate
```

Make sure TensorFlow is installed:
```
pip3 install tensorflow==2.17.0
```

Navigate to the work directory by activating the existing virtual environment for the project & clone the repo & try to interact with it:
```
cd /home/admin/databases/task_api_virtual_environment/
git clone git@github.com:m-kasim/challenge-task.git
git remote add origin git@github.com:m-kasim/challenge-task.git
git pull origin master
```

Finally, once all of this is done, you can try to run the project:
```
cd /home/admin/databases/task_api_virtual_environment/challenge-task

```

----

### Unit tests
To run all available `unit tests`, please execute:
```
python manage.py test
```

---

### Potential future improvements
Below is a list of features which have not been integrated due to time constraints, but can be added in future too:
- Integration with `Couchbase`: Every saved inference is saved to the database for faster access, if it is retrieved multiple times (the app would check for the result first in the DB, and would only query the `model` for inference if needed.
- Add `async` processing of incoming requests
- Add serving of TensorFlow model with `TensorFlow Serving` via `Docker` image
- Add request caching to the API calls and IP-rate limiting (or API key authentication filtering)
- Add ability to process entire PDF files (fetch abstract from the PDF and forward it to the model)
- Add `unit tests` per each API call
- Add a separate API call per each model (BERT single/multi-label and SciBERT single/multi-label)
- Store credentials in a proper `.env` directory/file
