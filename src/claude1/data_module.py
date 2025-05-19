import json
import uuid
from datetime import datetime

class DatabaseSchema:
    """
    Defines the database schema for the bruise detection system
    """
    def __init__(self):
        self.schema_version = "1.0.0"
        self.schema_description = "FHIR-compliant database schema for bruise detection"
        self.schema = self._build_schema()
    
    def _build_schema(self):
        """
        Build the complete database schema
        """
        schema = {
            "core_collections": self._build_core_collections(),
            "extension_collections": self._build_extension_collections(),
            "indexes": self._build_indexes(),
            "relationships": self._build_relationships(),
            "views": self._build_views()
        }
        return schema
    
    def _build_core_collections(self):
        """
        FHIR standard resource collections
        """
        collections = {
            "Patient": {
                "description": "Demographics and identifiers for patients",
                "fhir_resource_type": "Patient",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "active": {"type": "Boolean"},
                    "name": {"type": "Array<HumanName>"},
                    "telecom": {"type": "Array<ContactPoint>"},
                    "gender": {"type": "String", "enum": ["male", "female", "other", "unknown"]},
                    "birthDate": {"type": "Date"},
                    "address": {"type": "Array<Address>"},
                    "photo": {"type": "Array<Attachment>"},
                    "contact": {"type": "Array<Object>"},
                    "communication": {"type": "Array<Object>"},
                    "generalPractitioner": {"type": "Array<Reference>"},
                    "managingOrganization": {"type": "Reference"},
                    "link": {"type": "Array<Object>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/fitzpatrickSkinType",
                        "valueInteger": {"type": "Integer", "enum": [1, 2, 3, 4, 5, 6]}
                    },
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicCase",
                        "valueBoolean": {"type": "Boolean"}
                    }
                ]
            },
            "Practitioner": {
                "description": "Healthcare providers and staff",
                "fhir_resource_type": "Practitioner",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "active": {"type": "Boolean"},
                    "name": {"type": "Array<HumanName>"},
                    "telecom": {"type": "Array<ContactPoint>"},
                    "address": {"type": "Array<Address>"},
                    "gender": {"type": "String", "enum": ["male", "female", "other", "unknown"]},
                    "birthDate": {"type": "Date"},
                    "photo": {"type": "Array<Attachment>"},
                    "qualification": {"type": "Array<Object>"},
                    "communication": {"type": "Array<Object>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicCertification",
                        "valueCodeableConcept": {"type": "CodeableConcept"}
                    }
                ]
            },
            "DiagnosticReport": {
                "description": "Findings and interpretation of diagnostic tests",
                "fhir_resource_type": "DiagnosticReport",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "basedOn": {"type": "Array<Reference>"},
                    "status": {"type": "String", "enum": ["registered", "partial", "preliminary", "final", "amended", "corrected", "appended", "cancelled", "entered-in-error", "unknown"]},
                    "category": {"type": "Array<CodeableConcept>"},
                    "code": {"type": "CodeableConcept"},
                    "subject": {"type": "Reference"},
                    "encounter": {"type": "Reference"},
                    "effectiveDateTime": {"type": "DateTime"},
                    "effectivePeriod": {"type": "Period"},
                    "issued": {"type": "Instant"},
                    "performer": {"type": "Array<Reference>"},
                    "resultsInterpreter": {"type": "Array<Reference>"},
                    "specimen": {"type": "Array<Reference>"},
                    "result": {"type": "Array<Reference>"},
                    "imagingStudy": {"type": "Array<Reference>"},
                    "media": {"type": "Array<Object>"},
                    "conclusion": {"type": "String"},
                    "conclusionCode": {"type": "Array<CodeableConcept>"},
                    "presentedForm": {"type": "Array<Attachment>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id", "status", "code", "subject"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicDocumentation",
                        "valueBoolean": {"type": "Boolean"}
                    }
                ]
            },
            "ImagingStudy": {
                "description": "A set of imaging investigations",
                "fhir_resource_type": "ImagingStudy",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "status": {"type": "String", "enum": ["registered", "available", "cancelled", "entered-in-error", "unknown"]},
                    "modality": {"type": "Array<Coding>"},
                    "subject": {"type": "Reference"},
                    "encounter": {"type": "Reference"},
                    "started": {"type": "DateTime"},
                    "basedOn": {"type": "Array<Reference>"},
                    "referrer": {"type": "Reference"},
                    "interpreter": {"type": "Array<Reference>"},
                    "endpoint": {"type": "Array<Reference>"},
                    "numberOfSeries": {"type": "Integer"},
                    "numberOfInstances": {"type": "Integer"},
                    "procedureReference": {"type": "Reference"},
                    "procedureCode": {"type": "Array<CodeableConcept>"},
                    "location": {"type": "Reference"},
                    "reasonCode": {"type": "Array<CodeableConcept>"},
                    "reasonReference": {"type": "Array<Reference>"},
                    "note": {"type": "Array<Annotation>"},
                    "description": {"type": "String"},
                    "series": {"type": "Array<Object>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id", "status", "subject"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/alsParameter",
                        "valueObject": {
                            "wavelength": {"type": "Integer"},
                            "filter": {"type": "String"}
                        }
                    }
                ]
            },
            "Observation": {
                "description": "Measurements and assertions",
                "fhir_resource_type": "Observation",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "basedOn": {"type": "Array<Reference>"},
                    "partOf": {"type": "Array<Reference>"},
                    "status": {"type": "String", "enum": ["registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"]},
                    "category": {"type": "Array<CodeableConcept>"},
                    "code": {"type": "CodeableConcept"},
                    "subject": {"type": "Reference"},
                    "focus": {"type": "Array<Reference>"},
                    "encounter": {"type": "Reference"},
                    "effectiveDateTime": {"type": "DateTime"},
                    "effectivePeriod": {"type": "Period"},
                    "effectiveTiming": {"type": "Timing"},
                    "effectiveInstant": {"type": "Instant"},
                    "issued": {"type": "Instant"},
                    "performer": {"type": "Array<Reference>"},
                    "valueQuantity": {"type": "Quantity"},
                    "valueCodeableConcept": {"type": "CodeableConcept"},
                    "valueString": {"type": "String"},
                    "valueBoolean": {"type": "Boolean"},
                    "valueInteger": {"type": "Integer"},
                    "valueRange": {"type": "Range"},
                    "valueRatio": {"type": "Ratio"},
                    "valueSampledData": {"type": "SampledData"},
                    "valueTime": {"type": "Time"},
                    "valueDateTime": {"type": "DateTime"},
                    "valuePeriod": {"type": "Period"},
                    "dataAbsentReason": {"type": "CodeableConcept"},
                    "interpretation": {"type": "Array<CodeableConcept>"},
                    "note": {"type": "Array<Annotation>"},
                    "bodySite": {"type": "CodeableConcept"},
                    "method": {"type": "CodeableConcept"},
                    "specimen": {"type": "Reference"},
                    "device": {"type": "Reference"},
                    "referenceRange": {"type": "Array<Object>"},
                    "hasMember": {"type": "Array<Reference>"},
                    "derivedFrom": {"type": "Array<Reference>"},
                    "component": {"type": "Array<Object>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id", "status", "code"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/bruiseCharacteristics",
                        "valueObject": {
                            "size": {"type": "Quantity"},
                            "color": {"type": "String"},
                            "pattern": {"type": "String"},
                            "estimatedAge": {"type": "Object"}
                        }
                    }
                ]
            },
            "Media": {
                "description": "Photo, video, or audio content",
                "fhir_resource_type": "Media",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "identifier": {"type": "Array<Identifier>"},
                    "basedOn": {"type": "Array<Reference>"},
                    "partOf": {"type": "Array<Reference>"},
                    "status": {"type": "String", "enum": ["preparation", "in-progress", "not-done", "on-hold", "stopped", "completed", "entered-in-error", "unknown"]},
                    "type": {"type": "CodeableConcept"},
                    "modality": {"type": "CodeableConcept"},
                    "view": {"type": "CodeableConcept"},
                    "subject": {"type": "Reference"},
                    "encounter": {"type": "Reference"},
                    "createdDateTime": {"type": "DateTime"},
                    "createdPeriod": {"type": "Period"},
                    "issued": {"type": "Instant"},
                    "operator": {"type": "Reference"},
                    "reasonCode": {"type": "Array<CodeableConcept>"},
                    "bodySite": {"type": "CodeableConcept"},
                    "deviceName": {"type": "String"},
                    "device": {"type": "Reference"},
                    "height": {"type": "Integer"},
                    "width": {"type": "Integer"},
                    "frames": {"type": "Integer"},
                    "duration": {"type": "Decimal"},
                    "content": {"type": "Attachment"},
                    "note": {"type": "Array<Annotation>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id", "status", "content"],
                "extension_fields": [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/imagingParameters",
                        "valueObject": {
                            "exposureTime": {"type": "Integer"},
                            "iso": {"type": "Integer"},
                            "focalLength": {"type": "Decimal"},
                            "flash": {"type": "Boolean"},
                            "lightSource": {"type": "String"}
                        }
                    }
                ]
            },
            "AuditEvent": {
                "description": "Record of security relevant events",
                "fhir_resource_type": "AuditEvent",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "meta": {"type": "Object"},
                    "type": {"type": "Coding"},
                    "subtype": {"type": "Array<Coding>"},
                    "action": {"type": "String", "enum": ["C", "R", "U", "D", "E"]},
                    "period": {"type": "Period"},
                    "recorded": {"type": "Instant"},
                    "outcome": {"type": "String", "enum": ["0", "4", "8", "12"]},
                    "outcomeDesc": {"type": "String"},
                    "purposeOfEvent": {"type": "Array<CodeableConcept>"},
                    "agent": {"type": "Array<Object>"},
                    "source": {"type": "Object"},
                    "entity": {"type": "Array<Object>"},
                    "extension": {"type": "Array<Extension>"}
                },
                "required_fields": ["id", "type", "recorded", "agent", "source"],
                "extension_fields": []
            }
        }
        return collections
    
    def _build_extension_collections(self):
        """
        Custom extensions for bruise detection
        """
        collections = {
            "BruiseImage": {
                "description": "Extended media resource for bruise images",
                "extends": "Media",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "mediaId": {"type": "String", "reference": "Media.id"},
                    "lightSource": {"type": "String", "enum": ["white", "als_415nm", "als_450nm", "als_other"]},
                    "filterType": {"type": "String", "enum": ["none", "orange", "yellow", "red", "other"]},
                    "captureDevice": {"type": "String"},
                    "enhancementApplied": {"type": "Boolean"},
                    "enhancementParameters": {"type": "Object"},
                    "calibrationReference": {"type": "String"},
                    "originalImage": {"type": "Reference", "reference": "Media"},
                    "processingHistory": {"type": "Array<Object>"},
                    "imageHash": {"type": "String"},
                    "geolocation": {"type": "Object"},
                    "created": {"type": "DateTime"},
                    "createdBy": {"type": "Reference", "reference": "Practitioner"}
                },
                "required_fields": ["id", "mediaId", "lightSource"],
                "indexes": [
                    {"fields": ["mediaId"], "type": "unique"},
                    {"fields": ["lightSource"], "type": "index"},
                    {"fields": ["created"], "type": "index"}
                ]
            },
            "DetectionResult": {
                "description": "AI detection outputs for bruise images",
                "fields": {
                    "id": {"type": "String", "primary_key": True},
                    "bruiseImageId": {"type": "String", "reference": "BruiseImage.id"},
                    "imagingStudyId": {"type": "String", "reference": "ImagingStudy.id"},
                    "modelVersion": {"type": "String"},
                    "processingTimestamp": {"type": "DateTime"},
                    "detectionConfidence": {"type": "Decimal"},
                    "segmentationMap": {"type": "Reference", "reference": "Media"},
                    "bruiseArea": {"type": "Integer"},
                    "bruisePerimeter": {"type": "Integer"},
                    "colorAnalysis": {"type": "Object"},
                    "ageEstimation": {"type": "Object"},
                    "patternClassification": {"type": "Array<Object>"},
                    "suggestedFindings": {"type": "Array<CodeableConcept>"},
                    "validatedBy": {"type": "Reference", "reference": "Practitioner"},
                    "validationTimestamp": {"type": "DateTime"},
                    "validationResult": {"type": "String", "enum": ["accepted", "modified", "rejected"]},
                    "comments": {"type": "String"}
                },
                "required_fields": ["id", "bruiseImageId", "modelVersion", "processingTimestamp"],
                "indexes": [
                    {"fields": ["bruiseImageId"], "type": "index"},
                    {"fields": ["imagingStudyId"], "type": "index"},
                    {"fields": ["processingTimestamp"], "type": "index"}
                ]
            }
        }
        return collections
    
    def _build_indexes(self):
        """
        Define database indexes for optimal performance
        """
        indexes = {
            "Patient": [
                {"fields": ["identifier.value"], "type": "index"},
                {"fields": ["name.family", "name.given"], "type": "index"},
                {"fields": ["extension.url", "extension.valueInteger"], "type": "index", "sparse": True}
            ],
            "DiagnosticReport": [
                {"fields": ["subject"], "type": "index"},
                {"fields": ["code.coding.code"], "type": "index"},
                {"fields": ["imagingStudy"], "type": "index"},
                {"fields": ["effectiveDateTime"], "type": "index"}
            ],
            "ImagingStudy": [
                {"fields": ["subject"], "type": "index"},
                {"fields": ["started"], "type": "index"}
            ],
            "Observation": [
                {"fields": ["subject"], "type": "index"},
                {"fields": ["code.coding.code"], "type": "index"},
                {"fields": ["effectiveDateTime"], "type": "index"},
                {"fields": ["valueQuantity.value"], "type": "index", "sparse": True}
            ],
            "Media": [
                {"fields": ["subject"], "type": "index"},
                {"fields": ["createdDateTime"], "type": "index"}
            ],
            "AuditEvent": [
                {"fields": ["recorded"], "type": "index"},
                {"fields": ["agent.who.reference"], "type": "index"},
                {"fields": ["entity.what.reference"], "type": "index"}
            ]
        }
        return indexes
    
    def _build_relationships(self):
        """
        Define relationships between collections
        """
        relationships = [
            {
                "from": "DiagnosticReport",
                "to": "Patient",
                "type": "reference",
                "field": "subject",
                "cardinality": "many-to-one",
                "integrity": "cascade"
            },
            {
                "from": "DiagnosticReport",
                "to": "Practitioner",
                "type": "reference",
                "field": "performer",
                "cardinality": "many-to-many",
                "integrity": "restrict"
            },
            {
                "from": "DiagnosticReport",
                "to": "ImagingStudy",
                "type": "reference",
                "field": "imagingStudy",
                "cardinality": "many-to-many",
                "integrity": "cascade"
            },
            {
                "from": "DiagnosticReport",
                "to": "Observation",
                "type": "reference",
                "field": "result",
                "cardinality": "many-to-many",
                "integrity": "cascade"
            },
            {
                "from": "ImagingStudy",
                "to": "Patient",
                "type": "reference",
                "field": "subject",
                "cardinality": "many-to-one",
                "integrity": "cascade"
            },
            {
                "from": "Observation",
                "to": "Patient",
                "type": "reference",
                "field": "subject",
                "cardinality": "many-to-one",
                "integrity": "cascade"
            },
            {
                "from": "Media",
                "to": "Patient",
                "type": "reference",
                "field": "subject",
                "cardinality": "many-to-one",
                "integrity": "cascade"
            },
            {
                "from": "BruiseImage",
                "to": "Media",
                "type": "reference",
                "field": "mediaId",
                "cardinality": "one-to-one",
                "integrity": "cascade"
            },
            {
                "from": "DetectionResult",
                "to": "BruiseImage",
                "type": "reference",
                "field": "bruiseImageId",
                "cardinality": "many-to-one",
                "integrity": "cascade"
            },
            {
                "from": "DetectionResult",
                "to": "ImagingStudy",
                "type": "reference",
                "field": "imagingStudyId",
                "cardinality": "many-to-one",
                "integrity": "restrict"
            }
        ]
        return relationships
    
    def _build_views(self):
        """
        Define database views for common queries
        """
        views = {
            "PatientBruiseHistory": {
                "description": "Comprehensive view of patient bruise history",
                "query": """
                    SELECT 
                        p.id AS patientId,
                        p.name[0].family AS familyName,
                        p.name[0].given[0] AS givenName,
                        dr.id AS reportId,
                        dr.code.coding[0].display AS reportType,
                        dr.effectiveDateTime AS reportDate,
                        is.id AS imagingStudyId,
                        bi.id AS bruiseImageId,
                        bi.lightSource AS lightSource,
                        m.content.url AS imageUrl,
                        dr.conclusion AS conclusion,
                        det.detectionConfidence AS aiConfidence,
                        det.ageEstimation.estimatedHours AS estimatedAge
                    FROM Patient p
                    JOIN DiagnosticReport dr ON dr.subject.reference = CONCAT('Patient/', p.id)
                    JOIN ImagingStudy is ON is.id IN dr.imagingStudy[*].reference
                    JOIN Media m ON m.subject.reference = CONCAT('Patient/', p.id)
                    JOIN BruiseImage bi ON bi.mediaId = m.id
                    LEFT JOIN DetectionResult det ON det.bruiseImageId = bi.id
                    ORDER BY dr.effectiveDateTime DESC
                """
            },
            "BruiseDetectionMetrics": {
                "description": "Performance metrics for bruise detection",
                "query": """
                    SELECT 
                        det.modelVersion AS modelVersion,
                        DATE_TRUNC('day', det.processingTimestamp) AS date,
                        p.extension[?(@.url='https://bruise.gmu.edu/fhir/StructureDefinition/fitzpatrickSkinType')].valueInteger AS skinType,
                        COUNT(*) AS detectionCount,
                        AVG(det.detectionConfidence) AS avgConfidence,
                        STDDEV(det.detectionConfidence) AS stdDevConfidence,
                        COUNT(CASE WHEN det.validationResult = 'accepted' THEN 1 END) / COUNT(*) AS acceptanceRate
                    FROM DetectionResult det
                    JOIN BruiseImage bi ON bi.id = det.bruiseImageId
                    JOIN Media m ON m.id = bi.mediaId
                    JOIN Patient p ON p.id = SUBSTRING(m.subject.reference, 9)
                    GROUP BY modelVersion, date, skinType
                    ORDER BY date DESC, modelVersion, skinType
                """
            },
            "AuditTrail": {
                "description": "Complete audit trail for patient data",
                "query": """
                    SELECT 
                        ae.recorded AS timestamp,
                        ae.type.display AS eventType,
                        ae.action AS action,
                        ae.outcome AS outcome,
                        ae.agent[0].who.display AS actor,
                        ae.entity[0].what.reference AS resource,
                        ae.entity[0].type.display AS resourceType,
                        ae.source.site AS site
                    FROM AuditEvent ae
                    ORDER BY ae.recorded DESC
                """
            }
        }
        return views
    
    def get_collection_schema(self, collection_name):
        """
        Get schema for a specific collection
        """
        if collection_name in self.schema["core_collections"]:
            return self.schema["core_collections"][collection_name]
        elif collection_name in self.schema["extension_collections"]:
            return self.schema["extension_collections"][collection_name]
        return None
    
    def get_schema_as_json(self):
        """
        Return the complete schema as JSON
        """
        return json.dumps(self.schema, indent=2)
    
    def generate_sample_data(self, collection_name, count=1):
        """
        Generate sample data for a collection
        """
        schema = self.get_collection_schema(collection_name)
        if not schema:
            return None
        
        results = []
        for _ in range(count):
            record = {}
            
            # Add required fields
            for field in schema.get("required_fields", []):
                if field == "id":
                    record[field] = str(uuid.uuid4())
                elif field == "status":
                    # Get enum values if available
                    field_schema = schema["fields"].get(field, {})
                    enum_values = field_schema.get("enum", ["final"])
                    record[field] = enum_values[0]  # Choose first enum value
                elif field == "subject":
                    record[field] = {"reference": f"Patient/{str(uuid.uuid4())}"}
                elif field == "code":
                    record[field] = {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "86184-5",
                                "display": "Forensic injury evaluation"
                            }
                        ],
                        "text": "Bruise Detection Analysis"
                    }
                elif field == "content":
                    record[field] = {
                        "contentType": "image/jpeg",
                        "url": f"https://bruise.gmu.edu/images/{str(uuid.uuid4())}.jpg"
                    }
                elif field == "type":
                    record[field] = {
                        "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
                        "code": "rest",
                        "display": "RESTful Operation"
                    }
                elif field == "recorded":
                    record[field] = datetime.utcnow().isoformat()
                elif field == "agent":
                    record[field] = [
                        {
                            "type": {
                                "coding": [
                                    {
                                        "system": "http://terminology.hl7.org/CodeSystem/security-role-type",
                                        "code": "humanuser",
                                        "display": "Human User"
                                    }
                                ]
                            },
                            "who": {
                                "reference": f"Practitioner/{str(uuid.uuid4())}",
                                "display": "Dr. Example User"
                            }
                        }
                    ]
                elif field == "source":
                    record[field] = {
                        "site": "EAS-ID Mobile App",
                        "observer": {
                            "reference": "Device/mobileapp",
                            "display": "EAS-ID Mobile Application"
                        }
                    }
            
            # Add common fields
            if "fields" in schema and "meta" in schema["fields"]:
                record["meta"] = {
                    "versionId": "1",
                    "lastUpdated": datetime.utcnow().isoformat()
                }
            
            # Add extension fields if applicable
            if collection_name == "Patient":
                record["extension"] = [
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/fitzpatrickSkinType",
                        "valueInteger": 4
                    },
                    {
                        "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicCase",
                        "valueBoolean": True
                    }
                ]
            
            results.append(record)
        
        return results if count > 1 else results[0]


class FHIRDataModel:
    """
    Class for FHIR data modeling and validation
    """
    def __init__(self):
        self.version = "R4"
        self.base_url = "https://bruise.gmu.edu/fhir"
        self.supported_resources = [
            "Patient", "Practitioner", "DiagnosticReport", 
            "ImagingStudy", "Observation", "Media", "AuditEvent"
        ]
    
    def create_patient(self, patient_data):
        """
        Create a FHIR Patient resource
        """
        if "id" not in patient_data:
            patient_data["id"] = str(uuid.uuid4())
        
        patient = {
            "resourceType": "Patient",
            "id": patient_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "active": True
        }
        
        # Add name if provided
        if "name" in patient_data:
            if isinstance(patient_data["name"], list):
                patient["name"] = patient_data["name"]
            else:
                # Parse from string
                name_parts = patient_data["name"].split()
                if len(name_parts) > 1:
                    patient["name"] = [
                        {
                            "use": "official",
                            "family": name_parts[-1],
                            "given": name_parts[:-1]
                        }
                    ]
                else:
                    patient["name"] = [
                        {
                            "use": "official",
                            "given": name_parts
                        }
                    ]
        
        # Add other fields
        if "gender" in patient_data:
            patient["gender"] = patient_data["gender"]
        
        if "birthDate" in patient_data:
            patient["birthDate"] = patient_data["birthDate"]
        
        if "address" in patient_data:
            patient["address"] = patient_data["address"]
        
        # Add Fitzpatrick skin type extension if provided
        if "fitzpatrickSkinType" in patient_data:
            skin_type = int(patient_data["fitzpatrickSkinType"])
            if 1 <= skin_type <= 6:
                if "extension" not in patient:
                    patient["extension"] = []
                
                patient["extension"].append({
                    "url": "https://bruise.gmu.edu/fhir/StructureDefinition/fitzpatrickSkinType",
                    "valueInteger": skin_type
                })
        
        # Add forensic case extension if provided
        if "forensicCase" in patient_data:
            if "extension" not in patient:
                patient["extension"] = []
            
            patient["extension"].append({
                "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicCase",
                "valueBoolean": bool(patient_data["forensicCase"])
            })
        
        return patient
    
    def create_bruise_diagnostic_report(self, report_data):
        """
        Create a FHIR DiagnosticReport for bruise assessment
        """
        if "id" not in report_data:
            report_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "patientId" not in report_data:
            raise ValueError("patientId is required for DiagnosticReport")
        
        report = {
            "resourceType": "DiagnosticReport",
            "id": report_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "status": report_data.get("status", "final"),
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "IMG",
                            "display": "Imaging"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "86184-5",
                        "display": "Forensic injury evaluation"
                    }
                ],
                "text": "Bruise Detection Analysis"
            },
            "subject": {
                "reference": f"Patient/{report_data['patientId']}"
            },
            "effectiveDateTime": report_data.get("effectiveDateTime", datetime.utcnow().isoformat()),
            "issued": datetime.utcnow().isoformat()
        }
        
        # Add performer if provided
        if "performerId" in report_data:
            report["performer"] = [
                {
                    "reference": f"Practitioner/{report_data['performerId']}"
                }
            ]
        
        # Add imaging study references if provided
        if "imagingStudyIds" in report_data:
            report["imagingStudy"] = []
            for study_id in report_data["imagingStudyIds"]:
                report["imagingStudy"].append({
                    "reference": f"ImagingStudy/{study_id}"
                })
        
        # Add observation references if provided
        if "observationIds" in report_data:
            report["result"] = []
            for obs_id in report_data["observationIds"]:
                report["result"].append({
                    "reference": f"Observation/{obs_id}"
                })
        
        # Add media references if provided
        if "mediaIds" in report_data:
            report["media"] = []
            for i, media_id in enumerate(report_data["mediaIds"]):
                report["media"].append({
                    "comment": f"Bruise image {i+1}",
                    "link": {
                        "reference": f"Media/{media_id}"
                    }
                })
        
        # Add conclusion if provided
        if "conclusion" in report_data:
            report["conclusion"] = report_data["conclusion"]
        
        # Add forensic documentation extension if provided
        if "forensicDocumentation" in report_data:
            if "extension" not in report:
                report["extension"] = []
            
            report["extension"].append({
                "url": "https://bruise.gmu.edu/fhir/StructureDefinition/forensicDocumentation",
                "valueBoolean": bool(report_data["forensicDocumentation"])
            })
        
        return report
    
    def create_bruise_observation(self, observation_data):
        """
        Create a FHIR Observation for bruise documentation
        """
        if "id" not in observation_data:
            observation_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "patientId" not in observation_data:
            raise ValueError("patientId is required for Observation")
        
        observation = {
            "resourceType": "Observation",
            "id": observation_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "status": observation_data.get("status", "final"),
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "exam",
                            "display": "Examination"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11524-6",
                        "display": "Physical findings of Skin"
                    }
                ],
                "text": "Bruise Assessment"
            },
            "subject": {
                "reference": f"Patient/{observation_data['patientId']}"
            },
            "effectiveDateTime": observation_data.get("effectiveDateTime", datetime.utcnow().isoformat())
        }
        
        # Add body site if provided
        if "bodySite" in observation_data:
            observation["bodySite"] = {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": observation_data.get("bodySiteCode", "123456"),
                        "display": observation_data["bodySite"]
                    }
                ]
            }
        
        # Add bruise size if provided
        if "size" in observation_data:
            observation["valueQuantity"] = {
                "value": float(observation_data["size"]),
                "unit": "cm",
                "system": "http://unitsofmeasure.org",
                "code": "cm"
            }
        
        # Add bruise characteristics extension if provided
        bruise_characteristics = {}
        
        if "size" in observation_data:
            bruise_characteristics["size"] = {
                "value": float(observation_data["size"]),
                "unit": "cm"
            }
        
        if "color" in observation_data:
            bruise_characteristics["color"] = observation_data["color"]
        
        if "pattern" in observation_data:
            bruise_characteristics["pattern"] = observation_data["pattern"]
        
        if "estimatedAge" in observation_data:
            bruise_characteristics["estimatedAge"] = {
                "estimatedHours": int(observation_data["estimatedAge"]),
                "confidence": observation_data.get("ageConfidence", 0.75)
            }
        
        if bruise_characteristics:
            if "extension" not in observation:
                observation["extension"] = []
            
            observation["extension"].append({
                "url": "https://bruise.gmu.edu/fhir/StructureDefinition/bruiseCharacteristics",
                "valueObject": bruise_characteristics
            })
        
        return observation
    
    def create_imaging_study(self, study_data):
        """
        Create a FHIR ImagingStudy for bruise documentation
        """
        if "id" not in study_data:
            study_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "patientId" not in study_data:
            raise ValueError("patientId is required for ImagingStudy")
        
        study = {
            "resourceType": "ImagingStudy",
            "id": study_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "status": study_data.get("status", "available"),
            "subject": {
                "reference": f"Patient/{study_data['patientId']}"
            },
            "started": study_data.get("started", datetime.utcnow().isoformat()),
            "numberOfSeries": study_data.get("numberOfSeries", 1),
            "numberOfInstances": study_data.get("numberOfInstances", 1),
            "description": study_data.get("description", "Bruise documentation session")
        }
        
        # Add modality
        study["modality"] = [
            {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": "XC",
                "display": "External-camera Photography"
            }
        ]
        
        # Add ALS parameter extension if provided
        if "alsWavelength" in study_data or "alsFilter" in study_data:
            if "extension" not in study:
                study["extension"] = []
            
            study["extension"].append({
                "url": "https://bruise.gmu.edu/fhir/StructureDefinition/alsParameter",
                "valueObject": {
                    "wavelength": study_data.get("alsWavelength", 415),
                    "filter": study_data.get("alsFilter", "orange")
                }
            })
        
        return study
    
    def create_media(self, media_data):
        """
        Create a FHIR Media resource for bruise images
        """
        if "id" not in media_data:
            media_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "patientId" not in media_data:
            raise ValueError("patientId is required for Media")
        
        if "contentUrl" not in media_data:
            raise ValueError("contentUrl is required for Media")
        
        media = {
            "resourceType": "Media",
            "id": media_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "status": media_data.get("status", "completed"),
            "type": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/media-type",
                        "code": "photo",
                        "display": "Photo"
                    }
                ]
            },
            "subject": {
                "reference": f"Patient/{media_data['patientId']}"
            },
            "createdDateTime": media_data.get("createdDateTime", datetime.utcnow().isoformat()),
            "content": {
                "contentType": media_data.get("contentType", "image/jpeg"),
                "url": media_data["contentUrl"]
            }
        }
        
        # Add body site if provided
        if "bodySite" in media_data:
            media["bodySite"] = {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": media_data.get("bodySiteCode", "123456"),
                        "display": media_data["bodySite"]
                    }
                ]
            }
        
        # Add height and width if provided
        if "height" in media_data:
            media["height"] = int(media_data["height"])
        
        if "width" in media_data:
            media["width"] = int(media_data["width"])
        
        # Add imaging parameters extension if provided
        imaging_parameters = {}
        
        if "exposureTime" in media_data:
            imaging_parameters["exposureTime"] = int(media_data["exposureTime"])
        
        if "iso" in media_data:
            imaging_parameters["iso"] = int(media_data["iso"])
        
        if "focalLength" in media_data:
            imaging_parameters["focalLength"] = float(media_data["focalLength"])
        
        if "flash" in media_data:
            imaging_parameters["flash"] = bool(media_data["flash"])
        
        if "lightSource" in media_data:
            imaging_parameters["lightSource"] = media_data["lightSource"]
        
        if imaging_parameters:
            if "extension" not in media:
                media["extension"] = []
            
            media["extension"].append({
                "url": "https://bruise.gmu.edu/fhir/StructureDefinition/imagingParameters",
                "valueObject": imaging_parameters
            })
        
        return media
    
    def create_audit_event(self, audit_data):
        """
        Create a FHIR AuditEvent resource
        
        Parameters:
        - audit_data: Dictionary containing audit event data
        
        Returns:
        - audit_event: FHIR AuditEvent resource
        """
        if "id" not in audit_data:
            audit_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "type" not in audit_data:
            raise ValueError("Event type is required for AuditEvent")
        if "recorded" not in audit_data:
            raise ValueError("Recorded time is required for AuditEvent")
        if "agent" not in audit_data:
            raise ValueError("Agent information is required for AuditEvent")
        if "source" not in audit_data:
            raise ValueError("Source information is required for AuditEvent")
        
        audit_event = {
            "resourceType": "AuditEvent",
            "id": audit_data["id"],
            "meta": {
                "versionId": "1",
                "lastUpdated": datetime.utcnow().isoformat()
            },
            "type": {
                "system": "http://terminology.hl7.org/CodeSystem/audit-event-type",
                "code": audit_data["type"]["code"],
                "display": audit_data["type"]["display"]
            },
            "recorded": audit_data["recorded"],
            "agent": audit_data["agent"],
            "source": audit_data["source"]
        }
        
        # Add optional fields
        if "subtype" in audit_data:
            audit_event["subtype"] = audit_data["subtype"]
        
        if "action" in audit_data:
            # action: C=Create, R=Read, U=Update, D=Delete, E=Execute
            audit_event["action"] = audit_data["action"]
        
        if "outcome" in audit_data:
            # outcome: 0=Success, 4=Minor failure, 8=Serious failure, 12=Major failure
            audit_event["outcome"] = audit_data["outcome"]
        
        if "outcomeDesc" in audit_data:
            audit_event["outcomeDesc"] = audit_data["outcomeDesc"]
        
        if "entity" in audit_data:
            audit_event["entity"] = audit_data["entity"]
        
        return audit_event
    
    def create_bundle(self, resources, bundle_type="transaction"):
        """
        Create a FHIR Bundle containing multiple resources
        
        Parameters:
        - resources: List of FHIR resources
        - bundle_type: Type of bundle (transaction, batch, document, etc.)
        
        Returns:
        - bundle: FHIR Bundle resource
        """
        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": bundle_type,
            "timestamp": datetime.utcnow().isoformat(),
            "entry": []
        }
        
        for resource in resources:
            entry = {
                "resource": resource
            }
            
            # Add request details for transaction/batch bundles
            if bundle_type in ["transaction", "batch"]:
                if "resourceType" in resource and "id" in resource:
                    entry["request"] = {
                        "method": "PUT",
                        "url": f"{resource['resourceType']}/{resource['id']}"
                    }
            
            bundle["entry"].append(entry)
        
        return bundle
    
    def create_bruise_documentation_bundle(self, data):
        """
        Create a comprehensive FHIR bundle for bruise documentation
        
        Parameters:
        - data: Dictionary containing patient, report, observation, and media data
        
        Returns:
        - bundle: FHIR Bundle resource with all related resources
        """
        resources = []
        
        # Create Patient resource if patient data provided
        if "patient" in data:
            patient = self.create_patient(data["patient"])
            resources.append(patient)
            patient_id = patient["id"]
        else:
            patient_id = data.get("patientId")
            if not patient_id:
                raise ValueError("Patient data or patientId is required")
        
        # Create ImagingStudy if study data provided
        imaging_study_id = None
        if "imagingStudy" in data:
            study_data = data["imagingStudy"]
            study_data["patientId"] = patient_id
            imaging_study = self.create_imaging_study(study_data)
            resources.append(imaging_study)
            imaging_study_id = imaging_study["id"]
        
        # Create Media resources if media data provided
        media_ids = []
        if "media" in data:
            for media_item in data["media"]:
                media_item["patientId"] = patient_id
                media = self.create_media(media_item)
                resources.append(media)
                media_ids.append(media["id"])
        
        # Create Observation resources if observation data provided
        observation_ids = []
        if "observations" in data:
            for obs_item in data["observations"]:
                obs_item["patientId"] = patient_id
                observation = self.create_bruise_observation(obs_item)
                resources.append(observation)
                observation_ids.append(observation["id"])
        
        # Create DiagnosticReport if report data provided or if we have related resources
        if "diagnosticReport" in data or (imaging_study_id or media_ids or observation_ids):
            report_data = data.get("diagnosticReport", {})
            report_data["patientId"] = patient_id
            
            if imaging_study_id:
                report_data["imagingStudyIds"] = [imaging_study_id]
            
            if media_ids:
                report_data["mediaIds"] = media_ids
            
            if observation_ids:
                report_data["observationIds"] = observation_ids
            
            diagnostic_report = self.create_bruise_diagnostic_report(report_data)
            resources.append(diagnostic_report)
        
        # Create bundle with all resources
        return self.create_bundle(resources, "transaction")
    
    def create_bruise_image(self, image_data):
        """
        Create a BruiseImage extension resource (custom schema)
        
        Parameters:
        - image_data: Dictionary containing bruise image data
        
        Returns:
        - bruise_image: BruiseImage resource
        """
        if "id" not in image_data:
            image_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "mediaId" not in image_data:
            raise ValueError("mediaId is required for BruiseImage")
        if "lightSource" not in image_data:
            raise ValueError("lightSource is required for BruiseImage")
        
        bruise_image = {
            "id": image_data["id"],
            "mediaId": image_data["mediaId"],
            "lightSource": image_data["lightSource"],
            "created": image_data.get("created", datetime.utcnow().isoformat())
        }
        
        # Add optional fields
        if "filterType" in image_data:
            bruise_image["filterType"] = image_data["filterType"]
        
        if "captureDevice" in image_data:
            bruise_image["captureDevice"] = image_data["captureDevice"]
        
        if "enhancementApplied" in image_data:
            bruise_image["enhancementApplied"] = bool(image_data["enhancementApplied"])
        
        if "enhancementParameters" in image_data:
            bruise_image["enhancementParameters"] = image_data["enhancementParameters"]
        
        if "calibrationReference" in image_data:
            bruise_image["calibrationReference"] = image_data["calibrationReference"]
        
        if "originalImage" in image_data:
            bruise_image["originalImage"] = {
                "reference": f"Media/{image_data['originalImage']}"
            }
        
        if "processingHistory" in image_data:
            bruise_image["processingHistory"] = image_data["processingHistory"]
        
        if "imageHash" in image_data:
            bruise_image["imageHash"] = image_data["imageHash"]
        
        if "geolocation" in image_data:
            bruise_image["geolocation"] = image_data["geolocation"]
        
        if "createdBy" in image_data:
            bruise_image["createdBy"] = {
                "reference": f"Practitioner/{image_data['createdBy']}"
            }
        
        return bruise_image
    
    def create_detection_result(self, result_data):
        """
        Create a DetectionResult resource (custom schema)
        
        Parameters:
        - result_data: Dictionary containing detection result data
        
        Returns:
        - detection_result: DetectionResult resource
        """
        if "id" not in result_data:
            result_data["id"] = str(uuid.uuid4())
        
        # Validate required fields
        if "bruiseImageId" not in result_data:
            raise ValueError("bruiseImageId is required for DetectionResult")
        if "modelVersion" not in result_data:
            raise ValueError("modelVersion is required for DetectionResult")
        if "processingTimestamp" not in result_data:
            result_data["processingTimestamp"] = datetime.utcnow().isoformat()
        
        detection_result = {
            "id": result_data["id"],
            "bruiseImageId": result_data["bruiseImageId"],
            "modelVersion": result_data["modelVersion"],
            "processingTimestamp": result_data["processingTimestamp"],
            "detectionConfidence": result_data.get("detectionConfidence", 0.0)
        }
        
        # Add optional fields
        if "imagingStudyId" in result_data:
            detection_result["imagingStudyId"] = result_data["imagingStudyId"]
        
        if "segmentationMap" in result_data:
            detection_result["segmentationMap"] = {
                "reference": f"Media/{result_data['segmentationMap']}"
            }
        
        if "bruiseArea" in result_data:
            detection_result["bruiseArea"] = int(result_data["bruiseArea"])
        
        if "bruisePerimeter" in result_data:
            detection_result["bruisePerimeter"] = int(result_data["bruisePerimeter"])
        
        if "colorAnalysis" in result_data:
            detection_result["colorAnalysis"] = result_data["colorAnalysis"]
        
        if "ageEstimation" in result_data:
            detection_result["ageEstimation"] = result_data["ageEstimation"]
        
        if "patternClassification" in result_data:
            detection_result["patternClassification"] = result_data["patternClassification"]
        
        if "suggestedFindings" in result_data:
            detection_result["suggestedFindings"] = result_data["suggestedFindings"]
        
        if "validatedBy" in result_data:
            detection_result["validatedBy"] = {
                "reference": f"Practitioner/{result_data['validatedBy']}"
            }
        
        if "validationTimestamp" in result_data:
            detection_result["validationTimestamp"] = result_data["validationTimestamp"]
        
        if "validationResult" in result_data:
            detection_result["validationResult"] = result_data["validationResult"]
        
        if "comments" in result_data:
            detection_result["comments"] = result_data["comments"]
        
        return detection_result


class DataPipeline:
    """
    Class for managing data pipeline operations
    """
    def __init__(self, fhir_model):
        self.fhir_model = fhir_model
        self.schema = DatabaseSchema()
    
    def process_bruise_imaging_session(self, session_data):
        """
        Process a complete bruise imaging session
        
        Parameters:
        - session_data: Dictionary containing all session data
        
        Returns:
        - bundle: FHIR Bundle with all resources
        - processing_results: Dictionary with processing outcomes
        """
        resources = []
        processing_results = {
            "success": True,
            "resources_created": [],
            "errors": []
        }
        
        try:
            # Validate and create patient if needed
            if "patient" in session_data:
                patient = self.fhir_model.create_patient(session_data["patient"])
                resources.append(patient)
                processing_results["resources_created"].append({
                    "type": "Patient",
                    "id": patient["id"]
                })
                patient_id = patient["id"]
            else:
                patient_id = session_data.get("patientId")
                if not patient_id:
                    raise ValueError("Patient information is required")
            
            # Create imaging study
            study_data = session_data.get("imagingStudy", {})
            study_data["patientId"] = patient_id
            imaging_study = self.fhir_model.create_imaging_study(study_data)
            resources.append(imaging_study)
            processing_results["resources_created"].append({
                "type": "ImagingStudy",
                "id": imaging_study["id"]
            })
            
            # Process images
            media_resources = []
            bruise_images = []
            
            for image_data in session_data.get("images", []):
                # Create Media resource
                media_data = {
                    "patientId": patient_id,
                    "contentUrl": image_data.get("url"),
                    "contentType": image_data.get("contentType", "image/jpeg"),
                    "bodySite": image_data.get("bodySite"),
                    "createdDateTime": image_data.get("captureTime", datetime.utcnow().isoformat())
                }
                
                # Add technical parameters if available
                if "imageMetadata" in image_data:
                    metadata = image_data["imageMetadata"]
                    if "exposureTime" in metadata:
                        media_data["exposureTime"] = metadata["exposureTime"]
                    if "iso" in metadata:
                        media_data["iso"] = metadata["iso"]
                    if "focalLength" in metadata:
                        media_data["focalLength"] = metadata["focalLength"]
                    if "flash" in metadata:
                        media_data["flash"] = metadata["flash"]
                    if "width" in metadata:
                        media_data["width"] = metadata["width"]
                    if "height" in metadata:
                        media_data["height"] = metadata["height"]
                
                media = self.fhir_model.create_media(media_data)
                resources.append(media)
                media_resources.append(media["id"])
                
                # Create BruiseImage extension
                bruise_image_data = {
                    "mediaId": media["id"],
                    "lightSource": image_data.get("lightSource", "white"),
                    "filterType": image_data.get("filterType", "none"),
                    "captureDevice": image_data.get("captureDevice"),
                    "createdBy": session_data.get("practitionerId"),
                    "imageHash": image_data.get("hash")
                }
                
                bruise_image = self.fhir_model.create_bruise_image(bruise_image_data)
                bruise_images.append(bruise_image)
                processing_results["resources_created"].append({
                    "type": "BruiseImage",
                    "id": bruise_image["id"]
                })
            
            processing_results["resources_created"].extend([
                {"type": "Media", "id": mid} for mid in media_resources
            ])
            
            # Create observations for findings
            observation_resources = []
            
            for finding in session_data.get("findings", []):
                obs_data = {
                    "patientId": patient_id,
                    "bodySite": finding.get("bodySite"),
                    "size": finding.get("size"),
                    "color": finding.get("color"),
                    "pattern": finding.get("pattern"),
                    "estimatedAge": finding.get("ageHours"),
                    "ageConfidence": finding.get("ageConfidence"),
                    "effectiveDateTime": session_data.get("sessionTime", datetime.utcnow().isoformat())
                }
                
                observation = self.fhir_model.create_bruise_observation(obs_data)
                resources.append(observation)
                observation_resources.append(observation["id"])
                processing_results["resources_created"].append({
                    "type": "Observation",
                    "id": observation["id"]
                })
            
            # Create diagnostic report
            report_data = {
                "patientId": patient_id,
                "performerId": session_data.get("practitionerId"),
                "effectiveDateTime": session_data.get("sessionTime", datetime.utcnow().isoformat()),
                "status": "preliminary",
                "imagingStudyIds": [imaging_study["id"]],
                "mediaIds": media_resources,
                "observationIds": observation_resources,
                "conclusion": session_data.get("conclusion"),
                "forensicDocumentation": session_data.get("forensicDocumentation", True)
            }
            
            diagnostic_report = self.fhir_model.create_bruise_diagnostic_report(report_data)
            resources.append(diagnostic_report)
            processing_results["resources_created"].append({
                "type": "DiagnosticReport",
                "id": diagnostic_report["id"]
            })
            
            # Create audit event
            audit_data = {
                "type": {
                    "code": "rest",
                    "display": "RESTful Operation"
                },
                "subtype": [{
                    "system": "http://hl7.org/fhir/restful-interaction",
                    "code": "create",
                    "display": "create"
                }],
                "action": "C",
                "recorded": datetime.utcnow().isoformat(),
                "outcome": "0",
                "agent": [{
                    "type": {
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/security-role-type",
                            "code": "humanuser",
                            "display": "Human User"
                        }]
                    },
                    "who": {
                        "reference": f"Practitioner/{session_data.get('practitionerId', 'system')}",
                        "display": session_data.get("practitionerName", "System User")
                    },
                    "requestor": True
                }],
                "source": {
                    "site": "EAS-ID Mobile App",
                    "observer": {
                        "reference": "Device/mobileapp",
                        "display": "EAS-ID Mobile Application"
                    }
                },
                "entity": [{
                    "what": {
                        "reference": f"DiagnosticReport/{diagnostic_report['id']}"
                    },
                    "type": {
                        "system": "http://terminology.hl7.org/CodeSystem/audit-entity-type",
                        "code": "2",
                        "display": "System Object"
                    },
                    "role": {
                        "system": "http://terminology.hl7.org/CodeSystem/object-role",
                        "code": "3",
                        "display": "Report"
                    }
                }]
            }
            
            audit_event = self.fhir_model.create_audit_event(audit_data)
            resources.append(audit_event)
            processing_results["resources_created"].append({
                "type": "AuditEvent",
                "id": audit_event["id"]
            })
            
            # Create FHIR bundle
            bundle = self.fhir_model.create_bundle(resources, "transaction")
            
            # Store custom resources (BruiseImage, DetectionResult) separately
            processing_results["custom_resources"] = {
                "bruise_images": bruise_images
            }
            
            return bundle, processing_results
            
        except Exception as e:
            processing_results["success"] = False
            processing_results["errors"].append(str(e))
            return None, processing_results
    
    def validate_resource(self, resource, resource_type):
        """
        Validate a FHIR resource against schema
        
        Parameters:
        - resource: Resource to validate
        - resource_type: Type of resource
        
        Returns:
        - validation_results: Dictionary with validation outcomes
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Get schema for resource type
        schema = self.schema.get_collection_schema(resource_type)
        if not schema:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Unknown resource type: {resource_type}")
            return validation_results
        
        # Check required fields
        for field in schema.get("required_fields", []):
            if field not in resource:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required field: {field}")
        
        # Validate field types
        for field_name, field_def in schema.get("fields", {}).items():
            if field_name in resource:
                field_value = resource[field_name]
                expected_type = field_def.get("type")
                
                # Basic type validation
                if expected_type == "String" and not isinstance(field_value, str):
                    validation_results["warnings"].append(
                        f"Field {field_name} should be String, got {type(field_value).__name__}"
                    )
                elif expected_type == "Boolean" and not isinstance(field_value, bool):
                    validation_results["warnings"].append(
                        f"Field {field_name} should be Boolean, got {type(field_value).__name__}"
                    )
                elif expected_type == "Integer" and not isinstance(field_value, int):
                    validation_results["warnings"].append(
                        f"Field {field_name} should be Integer, got {type(field_value).__name__}"
                    )
                
                # Check enum values
                if "enum" in field_def and field_value not in field_def["enum"]:
                    validation_results["warnings"].append(
                        f"Field {field_name} value '{field_value}' not in allowed values: {field_def['enum']}"
                    )
        
        return validation_results
