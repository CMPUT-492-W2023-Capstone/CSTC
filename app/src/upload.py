import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


cred = credentials.Certificate('../credential/cmput492-cstc-2023-firebase-adminsdk-v0xys-66f2a6895d.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection(u'users').document(u'alovelace')
doc_ref.set({
    u'first': u'Ada',
    u'last': u'Lovelace',
    u'born': 1815
})
