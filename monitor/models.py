from django.db import models

# Create your models here.

class File(models.Model):
    file_ID = models.CharField(max_length=255)
    file_name = models.CharField(max_length=255)
    version = models.CharField(max_length=255)
    file_location = models.CharField(max_length=255)
    file_state = models.CharField(max_length=255)
    uploader = models.CharField(max_length=255)
    related_project = models.CharField(max_length=255)


class Users(models.Model):
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)

class role(models.Model):
    roles = models.CharField(max_length=255)

class user_role(models.Model):
    user_id = models.IntegerField()
    role_id = models.IntegerField()

class Vaults(models.Model):
    vault = models.CharField(max_length=255)

class vault_role(models.Model):
    vault_id = models.IntegerField()
    role_id = models.IntegerField()

class model_properties(models.Model):
    file_id = models.CharField(max_length=255)
    property_name = models.CharField(max_length=255)
    property_value = models.CharField(max_length=255)
    sequence_number = models.CharField(max_length=255)

class Model_structure_tree(models.Model):
    file_id = models.CharField(max_length=255)
    associated_file_id = models.CharField(max_length=255)
    sequence_number = models.CharField(max_length=255)



