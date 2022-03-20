# from webapp import db
#
# class Patient(db.Model):
#     patient_id = db.Column(db.Integer, primary_key=True)
#     triage_category = db.Column(db.String(50), nullable=False)
#     request = db.Column(db.String(500), nullable=False)
#     status = db.Column(db.String(50), nullable=False)
#     bed = db.relationship('Bed', backref='bed', lazy=True)
#
# class Bed(db.Model):
#     bed_id = db.Column(db.Integer, primary_key=True)
#     ward = db.Column(db.String(50), nullable=False)
#     patient = db.Column(db.Integer, db.ForeignKey('patient.patient_id'), nullable=True)
#
#

from webapp import db


class Bed(db.Model):
    bed_id = db.Column(db.Integer, primary_key=True)
    ward = db.Column(db.String(50), nullable=False)
    patient = db.relationship('Patient', backref='patient_bed',uselist=False, lazy=True)

class Patient(db.Model):
    patient_id = db.Column(db.Integer, primary_key=True)
    triage_category = db.Column(db.String(50), nullable=False)
    request = db.Column(db.String(500), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    bed = db.Column(db.Integer, db.ForeignKey('bed.bed_id'), nullable=True)


