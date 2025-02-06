from django.db import models
from django_mysql.models import ListCharField

# Create your models here.
class Intent(models.Model):
    tag = models.CharField(max_length=256)
    patterns = ListCharField(
        base_field=models.CharField(max_length=256),  # ✅ Define max_length
        size=None,
        max_length=256
    )
    responses = ListCharField(
        base_field=models.CharField(max_length=256),  # ✅ Define max_length
        size=None,
        max_length=256
    )
    
    def __str__(self):
        return self.tag
    
    class Meta:
        db_table = 'intents'